use ndarray::Array2;
use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    fs::{self, create_dir_all, rename, File},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant}
};
use tracing::{info, warn};
const MEL_MIN: f32 = -20.7232658369464104;
const MEL_DEQUANT_SCALE: f32 = 3.5572636272248159151598382543679e-4;
const MEL_QUANT_SCALE: f32 = 2811.1495373766991388922161119853;
macro_rules! defer {
    ($($stmt:stmt);* $(;)?) => {
        struct Defer<F: FnOnce()>(Option<F>);
        impl<F: FnOnce()> Drop for Defer<F> {
            fn drop(&mut self) { self.0.take().map(|f| f()); }
        }
        let _defer = Defer(Some(|| { $($stmt);* }));
    };
}
#[derive(Debug, Default)]
struct CrossProcessLockManager {
    lock_files: Mutex<HashMap<PathBuf, Arc<File>>>,
}
impl CrossProcessLockManager {
    fn get_lock_file(&self, path: &Path) -> Arc<File> {
        let lock_path = path.with_extension("lock");
        let mut lock_files = self.lock_files.lock().unwrap();
        if let Some(file) = lock_files.get(path) {
            return file.clone();
        }
        if let Some(parent) = lock_path.parent() {
            create_dir_all(parent).unwrap();
        }
        let file = File::options().read(true).write(true).create(true).open(&lock_path).unwrap();
        let file_arc = Arc::new(file);
        lock_files.insert(path.to_path_buf(), file_arc.clone());
        file_arc
    }
    fn acquire_shared(&self, path: &Path) {
        (&*self.get_lock_file(path)).lock_shared().unwrap();
    }
    fn acquire_exclusive(&self, path: &Path, timeout: Duration) {
        let lock_file = self.get_lock_file(path);
        let start = Instant::now();
        loop {
            if let Ok(()) = (&*lock_file).try_lock() {
                return;
            }
            if start.elapsed() >= timeout {
                panic!("Acquire exclusive lock timeout ({}ms): {:?}", timeout.as_millis(), path);
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
    fn release(&self, path: &Path) {
        (&*self.get_lock_file(path)).unlock().unwrap();
    }
}
#[derive(Debug, Default)]
pub struct CacheManager {
    lock_manager: CrossProcessLockManager,
}
impl CacheManager {
    fn validate_file_path(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            create_dir_all(parent).unwrap();
        }
    }
    pub fn load_features_cache(&self, path: &Path, force_gen: bool) -> Option<(Array2<f32>, f32, Vec<f32>)> {
        if force_gen || !path.exists() {
            return None;
        }
        self.lock_manager.acquire_shared(path);
        defer! { self.lock_manager.release(path); }
        let data = fs::read(path).map_err(|e| warn!("Open cache {} failed: {}", path.display(), e)).ok()?;
        if data.len() < 8 {
            warn!("Invalid cache file: {}", path.display());
            return None;
        }
        let cols = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let scale = f32::from_le_bytes(data[4..8].try_into().unwrap());
        let mel_bytes = 128 * cols * 2;
        let uv_bytes = (cols + 7) / 8;
        let expected_len = 8 + mel_bytes + uv_bytes;
        if data.len() != expected_len {
            warn!("Cache file size mismatch: expected {}, got {}", expected_len, data.len());
            return None;
        }
        let mut mel = Array2::zeros((128, cols));
        for (m, chunk) in mel.iter_mut().zip(data[8..8+mel_bytes].chunks_exact(2)) {
            *m = MEL_MIN + u16::from_le_bytes(chunk.try_into().unwrap()) as f32 * MEL_DEQUANT_SCALE;
        }
        let uv_compressed = &data[8+mel_bytes..];
        let mut uv = Vec::with_capacity(cols);
        for i in 0..cols {
            let bit = (uv_compressed[i / 8] >> (i % 8)) & 1;
            uv.push(if bit != 0 { 1.0 } else { 0.0 });
        }
        info!("Cache loaded: {}", path.display());
        Some((mel, scale, uv))
    }
    pub fn save_features_cache(&self, path: &Path, mel: &Array2<f32>, scale: f32, uv: &[f32]) {
        self.validate_file_path(path);
        self.lock_manager.acquire_exclusive(path, Duration::from_secs(5));
        defer! { self.lock_manager.release(path); }
        let cols = mel.ncols();
        let mel_bytes = 128 * cols * 2;
        let uv_bytes = (cols + 7) / 8;
        let mut buf = Vec::with_capacity(8 + mel_bytes + uv_bytes);
        buf.extend_from_slice(&(cols as u32).to_le_bytes());
        buf.extend_from_slice(&scale.to_le_bytes());
        buf.extend(mel.iter().flat_map(|&x| {
            let q = ((x - MEL_MIN) * MEL_QUANT_SCALE).round().clamp(0.0, 65535.0) as u16;
            q.to_le_bytes()
        }));
        let mut uv_compressed = vec![0u8; uv_bytes];
        for (i, &v) in uv.iter().enumerate() {
            if v > 0.5 {
                uv_compressed[i / 8] |= 1 << (i % 8);
            }
        }
        buf.extend_from_slice(&uv_compressed);
        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, &buf).unwrap();
        rename(&tmp_path, path).unwrap();
        info!("Features saved to: {}", path.display());
    }
}
pub static CACHE_MANAGER: Lazy<CacheManager> = Lazy::new(CacheManager::default);