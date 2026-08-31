use ebur128::{EbuR128, Mode};
use ndarray::{Array2, Axis, Zip, azip, s};
use crate::{audio::base_coeff::BASE_COEFF, consts::{NHV_CONFIG, SAMPLE_RATE, HOP_SIZE}, utils::{reflect_pad_1d, stft::{istft_core, stft_core}}};
pub fn pre_emphasis_base_tension(wave: &mut Vec<f32>, b: f32) {
    let orig_len = wave.len();
    let orig_max = wave.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let padded_len = ((orig_len + *HOP_SIZE - 1) / *HOP_SIZE) * *HOP_SIZE;
    wave.resize(padded_len, 0.0);
    let mut spec = stft_core(&wave); 
    let (_, freq_bins, n_frames) = spec.dim();
    let mut amp = Array2::zeros((freq_bins, n_frames));
    let re = spec.slice(s![0, .., ..]);
    let im = spec.slice(s![1, .., ..]);
    azip!((r in &re, i in &im, a in &mut amp) {
        *a = (r * r + i * i).sqrt().max(1e-9);
    });
    let mut orig_max_amp = 0.0;
    let mut f_max_amp = 0.0;
    for (j, mut bin) in amp.axis_iter_mut(Axis(0)).enumerate() {
        let scale = (b * BASE_COEFF[j]).clamp(-2.0, 2.0).exp();
        for v in bin.iter_mut() {
            if *v > orig_max_amp { orig_max_amp = *v; }
            *v *= scale;
            if *v > f_max_amp { f_max_amp = *v; }
        }
    }
    let gain = (orig_max_amp / f_max_amp) * ((-b / 15.0).clamp(0.0, 0.33) + 1.0);
    amp.mapv_inplace(|x| x * gain);
    Zip::indexed(amp.view())
        .for_each(|(j, i), &a| {
            let r = spec[[0, j, i]];
            let im = spec[[1, j, i]];
            let phase = im.atan2(r);
            spec[[0, j, i]] = a * phase.cos();
            spec[[1, j, i]] = a * phase.sin();
        });
    let filtered = istft_core(&spec, orig_len);
    let f_max_abs = filtered.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let gain2 = (orig_max / f_max_abs) * ((-b / 15.0).clamp(0.0, 0.33) + 1.0);
    wave.truncate(orig_len);
    for (w, f) in wave.iter_mut().zip(filtered.iter()) {
        *w = f * gain2;
    }
    wave.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
}
pub fn loudness_norm(wave: &mut Vec<f32>, target: f32, norm_strength: u8) {
    let orig_len = wave.len();
    let (mut val_start, mut val_end, mut need_restore) = (0, orig_len, false);
    if NHV_CONFIG.trim_silence {
        if 882 <= orig_len {
            let n_windows = (orig_len - 882) / 441 + 1;
            let energy_thresh = 10.0f32.powf(NHV_CONFIG.silence_threshold / 10.0) * 882 as f32;
            let mut sum_sq: f32 = wave[0..882].iter().map(|&x| x * x).sum();
            let mut start_idx = if sum_sq > energy_thresh { Some(0) } else { None };
            let mut end_idx = 0;
            for i in 1..n_windows {
                let prev_start = (i - 1) * 441;
                let new_start = i * 441;
                for j in prev_start..new_start {
                    sum_sq -= wave[j] * wave[j];
                }
                let prev_end = prev_start + 882;
                let new_end = new_start + 882;
                for j in prev_end..new_end {
                    sum_sq += wave[j] * wave[j];
                }
                if sum_sq > energy_thresh {
                    start_idx.get_or_insert(i);
                    end_idx = i;
                }
            }
            if let Some(s) = start_idx {
                val_start = s * 441;
                val_end = (end_idx * 441 + 5733).min(orig_len);
                need_restore = true;
            }
        }
    }
    let val_len = val_end - val_start;
    if val_len < 17640 {
        reflect_pad_1d(wave, 0, 17640 - val_len);
    }
    let measure_end = (val_start + val_len.max(17640)).min(wave.len());
    let audio_to_measure = &wave[val_start..measure_end];
    let mut ebu = EbuR128::new(1, SAMPLE_RATE, Mode::I)
        .expect("Failed to create EbuR128");
    ebu.add_frames_f32(audio_to_measure)
        .expect("Failed to add frames to EbuR128");
    let loudness_lkfs = ebu.loudness_global().unwrap_or(-150.0) as f32;
    let gain = 10.0f32.powf(
        (target - loudness_lkfs) * norm_strength as f32 * 0.0005,
    );
    if need_restore {
        let mut fade_len = 8820.min(val_len >> 2);
        if fade_len < 2 { fade_len = 0; }
        if fade_len > 0 {
            let fade_scale = 1.0 / (fade_len - 1) as f32;
            let vf = val_len - fade_len;
            for (i, x) in wave[val_start..val_end].iter_mut().enumerate() {
                let mut g = gain;
                if i >= vf {
                    g *= (i - vf) as f32 * fade_scale;
                } else if i < fade_len {
                    g *= i as f32 * fade_scale;
                }
                *x *= g;
            }
        } else {
            for x in &mut wave[val_start..val_end] {
                *x *= gain;
            }
        }
        wave[..val_start].fill(0.0);
        wave[val_end..].fill(0.0);
    } else {
        for x in &mut wave[val_start..val_end] {
            *x *= gain;
        }
    }
    wave.truncate(orig_len);
    wave.iter_mut().for_each(|x| *x = x.clamp(-1.0, 1.0));
}