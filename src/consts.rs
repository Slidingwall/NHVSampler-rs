pub const SAMPLE_RATE: u32 = 44100;
pub const FFT_SIZE: usize = 2048;
pub const ORIGIN_HOP_SIZE: usize = 128;
pub static IS_V3X: Lazy<bool> = Lazy::new(|| {
    let name = NHV_CONFIG.vocoder_path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    if name.contains("v3x") {
        true
    } else if name.contains("v3") {
        false
    } else {
        true
    }
});
pub static HOP_SIZE: Lazy<usize> = Lazy::new(|| if *IS_V3X { 512 } else { 256 });
pub static THOP: Lazy<f32> = Lazy::new(|| {
    *HOP_SIZE as f32 / SAMPLE_RATE as f32 * if *IS_V3X { 1.0 } else { 0.5 }
});
use ini::Ini;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::PathBuf;
#[derive(Debug, Clone, PartialEq)]
pub struct NHVConfig {
    pub vocoder_path: PathBuf,
    pub wave_norm: bool,
    pub trim_silence: bool,
    pub silence_threshold: f32,
    pub loop_mode: bool,
    pub peak_limit: f32,
    pub fill: usize,
    pub max_workers: usize,
    pub voiced_threshold: f32,
}
pub static NHV_CONFIG: Lazy<NHVConfig> = Lazy::new(|| load_nhv_config());
fn load_nhv_config() -> NHVConfig {
    let ini = match Ini::load_from_file("nhvconfig.ini") {
        Ok(ini) => ini,
        Err(_) => return NHVConfig::default(),
    };
    let def_sec: HashMap<String, String> = ini.section(None::<String>)
        .map(|props| props.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
        .unwrap_or_default();
    NHVConfig {
        vocoder_path: def_sec.get("vocoder_path").cloned().map(PathBuf::from)
            .unwrap_or(PathBuf::from("./model/nhv_v3x.onnx")),
        wave_norm: def_sec.get("wave_norm").and_then(|s| s.parse().ok())
            .unwrap_or(true),
        trim_silence: def_sec.get("trim_silence").and_then(|s| s.parse().ok())
            .unwrap_or(true),
        loop_mode: def_sec.get("loop_mode").and_then(|s| s.parse().ok())
            .unwrap_or(true),
        silence_threshold: def_sec.get("silence_threshold").and_then(|s| s.parse().ok())
            .unwrap_or(-52.0),
        peak_limit: def_sec.get("peak_limit").and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
        fill: def_sec.get("fill").and_then(|s| s.parse().ok())
            .unwrap_or(6),
        max_workers: def_sec.get("max_workers").and_then(|s| s.parse().ok())
            .unwrap_or(2),
        voiced_threshold: def_sec.get("voiced_threshold").and_then(|s| s.parse().ok())
            .unwrap_or(0.93),
    }
}
impl Default for NHVConfig {
    fn default() -> Self {
        Self {
            vocoder_path: PathBuf::from("./model/nhv_v3x.onnx"),
            wave_norm: true,
            trim_silence: true,
            silence_threshold: -52.0,
            loop_mode: true,
            peak_limit: 1.0,
            fill: 6,
            max_workers: 2,
            voiced_threshold: 0.93,
        }
    }
}
#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use super::*;
    #[test]
    fn test_default_config() {
        let default = NHVConfig::default();
        assert_eq!(
            default.vocoder_path,
            PathBuf::from("./model/nhv_v3x.onnx")
        );
        assert_eq!(default.wave_norm, true);
        assert_eq!(default.trim_silence, true);
        assert_eq!(default.silence_threshold, -52.0);
        assert_eq!(default.loop_mode, true);
        assert_eq!(default.peak_limit, 1.0);
        assert_eq!(default.fill, 6);
        assert_eq!(default.max_workers, 2);
        assert_eq!(default.voiced_threshold, 0.93);
    }
    #[test]
    fn test_global_config_init() {
        let cfg = &NHV_CONFIG;
        assert!(!cfg.vocoder_path.as_os_str().is_empty());
        assert!(cfg.silence_threshold.is_finite());
        assert!(cfg.peak_limit.is_finite());
        assert!(cfg.fill > 0);
        assert!(cfg.max_workers <= 32);
    }
    #[test]
    fn test_real_ini_load() {
        let ini_exists = Path::new("nhvconfig.ini").exists();
        let cfg = &NHV_CONFIG;
        if ini_exists {
            println!("Real nhvconfig.ini exists, verify parsed result is valid");
            assert!(!cfg.vocoder_path.as_os_str().is_empty());
        } else {
            println!("Real nhvconfig.ini does not exist, verify default config is returned");
            assert_eq!(**cfg, NHVConfig::default());
        }
    }
    #[test]
    fn test_parse_fault_tolerance() {
        let cfg = &NHV_CONFIG;
        assert!(cfg.silence_threshold.is_finite());
        assert!(cfg.peak_limit.is_finite());
        assert!(cfg.fill <= 100);
        assert!(cfg.max_workers >= 1 && cfg.max_workers <= 32);
    }
    #[test]
    fn test_vocoder_hop_consistency() {
        let is_v3x = *IS_V3X;
        let hop = *HOP_SIZE;
        let thop = *THOP;
        assert!(hop == 256 || hop == 512, "HOP_SIZE must be 256 (v3) or 512 (v3x)");
        assert_eq!(hop, if is_v3x { 512 } else { 256 });
        let expected = hop as f32 / SAMPLE_RATE as f32 * if is_v3x { 1.0 } else { 0.5 };
        assert!((thop - expected).abs() < 1e-6, "thop must equal hop/SR times 1.0 for v3x or 0.5 for v3");
    }
}