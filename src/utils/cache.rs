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
use crate::consts::ORIGIN_HOP_SIZE;
pub const FEATURE_EXT: &str = ".nhv.bin";
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
    pub fn load_features_cache(&self, path: &Path) -> Option<(Array2<f32>, f32, Vec<f32>)> {
        if !path.exists() {
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
pub(crate) fn load_frq_f0(path: &Path, origin_frames: usize) -> Option<Vec<f32>> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 40 || &bytes[0..8] != b"FREQ0003" {
        return None;
    }
    let spf = i32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let nframes = i32::from_le_bytes(bytes[36..40].try_into().ok()?) as usize;
    if nframes == 0 || spf == 0 || origin_frames == 0 || bytes.len() != 40 + nframes * 16 {
        return None;
    }
    let spf_f = spf as f32;
    let mut out = Vec::with_capacity(origin_frames);
    for k in 0..origin_frames {
        let fr = ((k as f32 * ORIGIN_HOP_SIZE as f32 / spf_f).round().clamp(0.0, (nframes - 1) as f32)) as usize;
        let raw = f32::from_le_bytes(bytes[40 + fr * 16 + 12..40 + fr * 16 + 16].try_into().ok()?);
        let f0 = raw * 100.0;
        out.push(if f0.is_finite() && f0 > 0.0 { f0 } else { 0.0 });
    }
    Some(out)
}
#[cfg(test)]
mod frq_tests {
    use super::*;
    fn make_frq(f0s_hz: &[f32], samples_per_frq: i32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"FREQ0003");
        b.extend_from_slice(&samples_per_frq.to_le_bytes());
        b.extend_from_slice(&[0u8; 24]);
        b.extend_from_slice(&(f0s_hz.len() as i32).to_le_bytes());
        for &hz in f0s_hz {
            for _ in 0..3 {
                b.extend_from_slice(&0.0f32.to_le_bytes());
            }
            b.extend_from_slice(&((hz / 100.0) as f32).to_le_bytes());
        }
        b
    }
    fn write_tmp(name: &str, bytes: &[u8]) -> PathBuf {
        let p = std::env::temp_dir().join(name);
        fs::write(&p, bytes).unwrap();
        p
    }
    #[test]
    fn load_frq_f0_known_values() {
        let p = write_tmp("nhv_frq_known.frq", &make_frq(&[200.0, 0.0, 440.0], 128));
        let out = load_frq_f0(&p, 3).expect("load");
        assert_eq!(out.len(), 3);
        assert!((out[0] - 200.0).abs() < 1.0, "frame0 f0={}", out[0]);
        assert_eq!(out[1], 0.0, "f0=0 -> unvoiced (f0=0)");
        assert!((out[2] - 440.0).abs() < 1.0, "frame2 f0={}", out[2]);
        fs::remove_file(&p).ok();
    }
    #[test]
    fn load_frq_f0_rejects_non_magic() {
        let p = write_tmp("nhv_frq_badmagic.frq", b"NOTFREQ0........");
        assert!(load_frq_f0(&p, 3).is_none());
        fs::remove_file(&p).ok();
    }
    #[test]
    fn load_frq_f0_rejects_bad_size() {
        let mut b = make_frq(&[200.0, 300.0], 256);
        b.pop();
        let p = write_tmp("nhv_frq_badsize.frq", &b);
        assert!(load_frq_f0(&p, 2).is_none());
        fs::remove_file(&p).ok();
    }
    #[test]
    fn load_frq_f0_zero_frames_rejected() {
        let p = write_tmp("nhv_frq_zero.frq", &make_frq(&[], 256));
        assert!(load_frq_f0(&p, 1).is_none());
        fs::remove_file(&p).ok();
    }
    #[test]
    fn load_frq_f0_timeshift_resamples_to_origin() {
        let f0s: Vec<f32> = (0..121).map(|i| 250.0 + i as f32).collect();
        let p = write_tmp("nhv_frq_shift.frq", &make_frq(&f0s, 256));
        let out = load_frq_f0(&p, 242).expect("load");
        assert_eq!(out.len(), 242);
        assert!(out[0] > 0.0, "all positive f0 -> all > 0");
        assert!(out[241] > 0.0, "all positive f0 -> all > 0");
        assert!(out.iter().all(|&v| v > 0.0), "every frame should be voiced (f0>0)");
        fs::remove_file(&p).ok();
    }
    #[test]
    fn load_frq_f0_real_provided_file() {
        let dir = std::env::var("NHV_FRQ_TEST_DIR").unwrap_or_default();
        if dir.is_empty() {
            eprintln!("skip: set NHV_FRQ_TEST_DIR to enable the real-frq test");
            return;
        }
        let p = Path::new(&dir).join("src-7f064a5e-510f4d17-09506d85_wav.frq");
        if !p.exists() {
            eprintln!("skip: provided frq not present");
            return;
        }
        let out = load_frq_f0(&p, 162).expect("load provided frq");
        let voiced: Vec<f32> = out.iter().copied().filter(|&v| v > 0.0).collect();
        assert!(!voiced.is_empty(), "should contain voiced frames");
        let mn = voiced.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = voiced.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("provided frq f0 range = {:.1}..{:.1} Hz ({} voiced frames)", mn, mx, voiced.len());
        assert!(mn >= 40.0 && mx <= 1200.0, "f0 should be within singing range");
    }
}