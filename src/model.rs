pub mod nhv;
use std::sync::{Mutex, atomic::{AtomicUsize, Ordering}};
use once_cell::sync::OnceCell;
use crate::consts::NHV_CONFIG;
use crate::model::{nhv::NHVLoader};
static VOCODER_POOL: OnceCell<Vec<Mutex<NHVLoader>>> = OnceCell::new();
static NEXT_VOCODER: AtomicUsize = AtomicUsize::new(0);
pub fn initialize_models(max_workers: usize) {
    if !NHV_CONFIG.vocoder_path.exists() {
        tracing::error!("NHV model not found at: {}", NHV_CONFIG.vocoder_path.display());
    }
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let pool_size = max_workers.min(cpu_cores).max(1);
    tracing::info!("Creating model pool with size = {}", pool_size);
    let vocoder_pool = (0..pool_size)
        .map(|_| Mutex::new(NHVLoader::new(&NHV_CONFIG.vocoder_path)))
        .collect();
    VOCODER_POOL.set(vocoder_pool).unwrap();
    tracing::info!("All models initialized successfully.");
}
pub fn get_vocoder() -> &'static Mutex<NHVLoader> {
    let pool = VOCODER_POOL.get().expect("Vocoder pool not initialized");
    let idx = NEXT_VOCODER.fetch_add(1, Ordering::Relaxed) % pool.len();
    &pool[idx]
}