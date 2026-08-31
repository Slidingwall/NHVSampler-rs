use std::path::Path;
use crate::utils::cache::load_frq_f0;
use crate::utils::llsm::read_llsm_vuv;
const VUV_F0_MIN: f32 = 70.0;
const VUV_WIN_MUL: usize = 2;
const VUV_F0_MAX: f32 = 1100.0;
const VUV_HP_HZ: f32 = 40.0;
const VUV_LAG_DECIM: usize = 8;
pub fn vuv(wave: &[f32], in_file: &Path, sr: u32, hop: usize) -> Vec<f32> {
    let n_frames = (wave.len() + hop - 1) / hop;
    let stem = in_file.file_name().unwrap().to_string_lossy();
    let llsm_p = in_file.with_file_name(format!("{stem}.llsm"));
    let llsm_p = if llsm_p.exists() { llsm_p } else { in_file.with_extension("llsm") };
    if let Some(uv) = read_llsm_vuv(&llsm_p, n_frames, hop, sr) {
        return uv;
    }
    let frq_p = in_file.with_file_name(format!("{stem}.frq"));
    let frq_p = if frq_p.exists() { frq_p } else { in_file.with_extension("frq") };
    if let Some(f0) = load_frq_f0(&frq_p, n_frames) {
        return f0.iter().map(|&f| if f > 0.0 { 0.0 } else { 1.0 }).collect();
    }
    let sr_f = sr as f32;
    let mut uv = vec![1.0f32; n_frames];
    let sil_thr = 10f32.powf(crate::consts::NHV_CONFIG.silence_threshold / 20.0)
        * wave.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let lag_min = (sr_f / VUV_F0_MAX).floor() as usize;
    let lag_max = (sr_f / VUV_F0_MIN).ceil() as usize;
    let win_max = lag_max * VUV_WIN_MUL;
    let hp_a = (-2.0 * std::f32::consts::PI * VUV_HP_HZ / sr_f).exp();
    let mut hp = vec![0.0f32; win_max];
    let mut e = vec![0.0f32; win_max];
    for frame in 0..n_frames {
        let start = frame * hop;
        let w = win_max.min(wave.len() - start);
        if w <= lag_max {
            continue;
        }
        let win = &wave[start..start + w];
        if (win.iter().map(|&x| x * x).sum::<f32>() / w as f32).sqrt() < sil_thr {
            continue;
        }
        let mut prev_x = 0.0f32;
        let mut prev_y = 0.0f32;
        for (i, &x) in win.iter().enumerate() {
            prev_y = x - prev_x + hp_a * prev_y;
            prev_x = x;
            hp[i] = prev_y;
        }
        let mut acc = 0.0f32;
        for i in 0..w {
            acc += hp[i] * hp[i];
            e[i] = acc;
        }
        let d = VUV_LAG_DECIM;
        let mut best_tau = lag_min;
        let mut best_r = 0.0f32;
        for tau in (lag_min + d - 1) / d..=(lag_max / d).min((w - 1) / d) {
            let t = tau * d;
            let tw = w - t;
            let (a, b) = (&hp[..tw], &hp[t..w]);
            let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let mut i = 0usize;
            while i + 4 <= tw {
                s0 += a[i] * b[i];
                s1 += a[i + 1] * b[i + 1];
                s2 += a[i + 2] * b[i + 2];
                s3 += a[i + 3] * b[i + 3];
                i += 4;
            }
            while i < tw {
                s0 += a[i] * b[i];
                i += 1;
            }
            let r = ((s0 + s1) + (s2 + s3)) / (e[tw - 1] * (e[w - 1] - e[t - 1])).sqrt();
            if r > best_r {
                best_r = r;
                best_tau = t;
            }
        }
        for tau in (best_tau - d).max(lag_min)..=(best_tau + d).min(lag_max).min(w - 1) {
            let tw = w - tau;
            let (a, b) = (&hp[..tw], &hp[tau..w]);
            let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let mut i = 0usize;
            while i + 4 <= tw {
                s0 += a[i] * b[i];
                s1 += a[i + 1] * b[i + 1];
                s2 += a[i + 2] * b[i + 2];
                s3 += a[i + 3] * b[i + 3];
                i += 4;
            }
            while i < tw {
                s0 += a[i] * b[i];
                i += 1;
            }
            let r = ((s0 + s1) + (s2 + s3)) / (e[tw - 1] * (e[w - 1] - e[tau - 1])).sqrt();
            if r > best_r {
                best_r = r;
            }
        }
        uv[frame] = if best_r > crate::consts::NHV_CONFIG.voiced_threshold {
            0.0
        } else {
            1.0
        };
    }
    let mut smoothed = vec![0.0f32; n_frames];
    for i in 0..n_frames {
        let lo = i.saturating_sub(2);
        let hi = (i + 2).min(n_frames - 1);
        let ones = (lo..=hi).filter(|&j| uv[j] > 0.5).count();
        smoothed[i] = if ones >= (hi - lo + 2) / 2 { 1.0 } else { 0.0 };
    }
    for i in 0..n_frames {
        let prev = if i == 0 { smoothed[i] } else { smoothed[i - 1] };
        let next = if i + 1 == n_frames { smoothed[i] } else { smoothed[i + 1] };
        uv[i] = if smoothed[i] != prev && smoothed[i] != next { prev } else { smoothed[i] };
    }
    uv
}
#[cfg(test)]
mod tests {
    use super::*;
    const SR: u32 = 44100;
    const HOP: usize = 128;
    const NO_ANALYSIS: &str = "__vuv_no_analysis__/note.wav";
    fn build_voiced_unvoiced() -> Vec<f32> {
        let n = SR as usize;
        let mut w = vec![0.0f32; n];
        for i in 0..(n / 2) {
            let t = i as f32 / SR as f32;
            w[i] = 0.3 * (2.0 * std::f32::consts::PI * 200.0 * t).sin();
        }
        let mut rng = 12345u32;
        for i in (n / 2)..n {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (rng >> 8) as f32 / ((1u32 << 24) as f32) - 0.5;
            w[i] = 0.3 * u;
        }
        w
    }
    #[test]
    fn test_voiced_vs_unvoiced_separation() {
        let wave = build_voiced_unvoiced();
        let uv = vuv(&wave, Path::new(NO_ANALYSIS), SR, HOP);
        let half = uv.len() / 2;
        let voiced_first = uv[..half].iter().filter(|&&v| v < 0.5).count();
        let voiced_second = uv[half..].iter().filter(|&&v| v < 0.5).count();
        println!("voiced first-half={} second-half={} (of {})", voiced_first, voiced_second, half);
        assert!(voiced_first as f32 / half as f32 > 0.9, "voiced segment should be almost all voiced");
        assert!(voiced_second as f32 / (half as f32) < 0.1, "unvoiced (white noise) segment should be almost all unvoiced");
    }
    #[test]
    fn test_aspirated_lowpass_is_unvoiced() {
        let n = SR as usize;
        let mut w = vec![0.0f32; n];
        let mut y = 0.0f32;
        let mut rng = 777u32;
        for i in 0..n {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = (rng >> 8) as f32 / ((1u32 << 24) as f32) - 0.5;
            y = 0.95 * y + 0.05 * x;
            w[i] = 0.3 * y;
        }
        let uv = vuv(&w, Path::new(NO_ANALYSIS), SR, HOP);
        let voiced = uv.iter().filter(|&&v| v < 0.5).count();
        assert!(voiced as f32 / (uv.len() as f32) < 0.1, "aspirated low-pass noise should be unvoiced");
    }
    #[test]
    fn test_falsetto_noisy_harmonic_is_voiced() {
        let n = SR as usize;
        let mut w = vec![0.0f32; n];
        let mut rng = 999u32;
        for i in 0..n {
            let t = i as f32 / SR as f32;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (rng >> 8) as f32 / ((1u32 << 24) as f32) - 0.5;
            w[i] = 0.25 * (2.0 * std::f32::consts::PI * 600.0 * t).sin() + 0.15 * noise;
        }
        let uv = vuv(&w, Path::new(NO_ANALYSIS), SR, HOP);
        let voiced = uv.iter().filter(|&&v| v < 0.5).count();
        println!("falsetto voiced ratio = {:.2}", voiced as f32 / uv.len() as f32);
        assert!(voiced as f32 / uv.len() as f32 > 0.85, "falsetto (harmonic + noise) should be voiced");
    }
    #[test]
    fn test_low_pitches_survive_the_high_pass() {
        let n = 2 * SR as usize;
        let mut rng = 1234u32;
        for f0 in [80.0f32, 100.0, 150.0, 200.0, 440.0, 800.0, 1000.0] {
            let mut w = vec![0.0f32; n];
            for i in 0..n {
                let t = i as f32 / SR as f32;
                w[i] = 0.25 * (2.0 * std::f32::consts::PI * f0 * t).sin();
            }
            let uv = vuv(&w, Path::new(NO_ANALYSIS), SR, HOP);
            let voiced = uv.iter().filter(|&&v| v < 0.5).count();
            assert!(
                voiced as f32 / uv.len() as f32 > 0.95,
                "{f0} Hz pure tone lost to the high-pass (voiced ratio {})",
                voiced as f32 / uv.len() as f32
            );
            let mut wb = vec![0.0f32; n];
            for i in 0..n {
                let t = i as f32 / SR as f32;
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (rng >> 8) as f32 / ((1u32 << 24) as f32) - 0.5;
                wb[i] = 0.25 * (2.0 * std::f32::consts::PI * f0 * t).sin() + 0.10 * noise;
            }
            let uv = vuv(&wb, Path::new(NO_ANALYSIS), SR, HOP);
            let voiced = uv.iter().filter(|&&v| v < 0.5).count();
            assert!(
                voiced as f32 / uv.len() as f32 > 0.95,
                "{f0} Hz breathy tone lost to the high-pass (voiced ratio {})",
                voiced as f32 / uv.len() as f32
            );
        }
    }
    #[test]
    fn test_breathy_long_vowel_stays_voiced() {
        let n = 3 * SR as usize;
        let mut w = vec![0.0f32; n];
        let mut rng = 4242u32;
        for i in 0..n {
            let t = i as f32 / SR as f32;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (rng >> 8) as f32 / ((1u32 << 24) as f32) - 0.5;
            w[i] = 0.25 * (2.0 * std::f32::consts::PI * 200.0 * t).sin() + 0.10 * noise;
        }
        let uv = vuv(&w, Path::new(NO_ANALYSIS), SR, HOP);
        let voiced = uv[..1000].iter().filter(|&&v| v < 0.5).count();
        assert!(voiced as f32 > 950.0, "breathy long vowel must stay voiced");
    }
    #[test]
    fn test_heavy_breath_is_reported_unvoiced() {
        let n = 3 * SR as usize;
        let mut w = vec![0.0f32; n];
        let mut rng = 4242u32;
        for i in 0..n {
            let t = i as f32 / SR as f32;
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (rng >> 8) as f32 / ((1u32 << 24) as f32) - 0.5;
            w[i] = 0.15 * (2.0 * std::f32::consts::PI * 200.0 * t).sin() + 0.30 * noise;
        }
        let uv = vuv(&w, Path::new(NO_ANALYSIS), SR, HOP);
        let voiced = uv[..1000].iter().filter(|&&v| v < 0.5).count();
        assert_eq!(voiced, 0, "SNR -6 dB is below the detector's voicing floor");
    }
}