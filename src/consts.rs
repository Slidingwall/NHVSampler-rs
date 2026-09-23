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
pub const FORMANT_HR: [f32; 128] = [
0.0244245574, 0.0248635931, 0.0253144694, 0.0257776121, 0.0262534659, 0.0267424959, 0.0272451884, 0.0277620519,
0.0282936187, 0.0288404462, 0.0294031179, 0.0299822452, 0.0305784691, 0.0311924616, 0.0318249278, 0.0324766074,
0.0331482773, 0.0338407536, 0.0345548936, 0.035291599, 0.0360518181, 0.0368365486, 0.0376468413, 0.038483803,
0.0393485998, 0.0402424617, 0.041166686, 0.0421226421, 0.0431117762, 0.0441356164, 0.0451957781, 0.0462939702,
0.0474320013, 0.0486117867, 0.0498646329, 0.0512194251, 0.0526724132, 0.0542334345, 0.0559136133, 0.0577255702,
0.0596836717, 0.0618043302, 0.0641063644, 0.0666114376, 0.0693445899, 0.0723348903, 0.0756162397, 0.0792283637,
0.0832180505, 0.0876407015, 0.0925622864, 0.0980618267, 0.10423457, 0.111196073, 0.119087499, 0.128082518,
0.138396377, 0.150297852, 0.164125093, 0.180306631, 0.199389148, 0.222073796, 0.249262529, 0.282114121,
0.322103989, 0.371067379, 0.431170727, 0.504680207, 0.593252123, 0.696269222, 0.807733175, 0.912269871,
0.984130954, 0.996214109, 0.938517834, 0.827799576, 0.695127327, 0.566584601, 0.455486045, 0.365102158,
0.293687885, 0.237887387, 0.194329128, 0.160171549, 0.133186499, 0.111681311, 0.0943859019, 0.0803494343,
0.0688577826, 0.0593712771, 0.0514789132, 0.0448650094, 0.039284973, 0.0345476387, 0.0305023267, 0.0270292956,
0.0240326446, 0.0214349952, 0.0191734731, 0.0171966487, 0.0154621883, 0.0139350379, 0.0125860083, 0.011390665,
0.0103284504, 0.00938198725, 0.00853652045, 0.00777946815, 0.00710005848, 0.00648903401, 0.00593840997, 0.0054412756,
0.00499163014, 0.0045842468, 0.00421455942, 0.00387856772, 0.00357275776, 0.00329403486, 0.00303966691, 0.00280723623,
0.0025945986, 0.00239984825, 0.00222128786, 0.00205740278, 0.00190683886, 0.00176838322, 0.00164094765, 0.00152355415
];
