use crate::{consts::{FFT_SIZE, HOP_SIZE}, utils::hann_window::HANN_WINDOW};
use ndarray::{Array3, Axis, parallel::prelude::*, s};
use once_cell::sync::Lazy;
use phastft::{c2r_fft_f32_with_planner, planner::PlannerR2c32, r2c_fft_f32_with_planner};
static FFT_PLANNER: Lazy<PlannerR2c32> = Lazy::new(|| PlannerR2c32::new(FFT_SIZE));
static HANN_FFT_SCALE: Lazy<[f32; FFT_SIZE]> = Lazy::new(|| {
    let mut a = [0.0; FFT_SIZE];
    for (i, v) in a.iter_mut().enumerate() {
        *v = HANN_WINDOW[i] / FFT_SIZE as f32;
    }
    a
});
static HANN_SQ: Lazy<[f32; FFT_SIZE]> = Lazy::new(|| {
    let mut a = [0.0; FFT_SIZE];
    for (i, v) in a.iter_mut().enumerate() {
        *v = HANN_WINDOW[i] * HANN_WINDOW[i];
    }
    a
});
thread_local! {
    static REAL_BUF: std::cell::RefCell<[f32; FFT_SIZE]> = std::cell::RefCell::new([0.0; FFT_SIZE]);
    static RE_BUF: std::cell::RefCell<[f32; 1025]> = std::cell::RefCell::new([0.0; 1025]);
    static IM_BUF: std::cell::RefCell<[f32; 1025]> = std::cell::RefCell::new([0.0; 1025]);
}
pub fn stft_core(signal: &[f32]) -> Array3<f32> {
    let freq_bins = FFT_SIZE / 2 + 1;
    let n_frames = (signal.len() + *HOP_SIZE - 1) / *HOP_SIZE;
    let planner = &*FFT_PLANNER;
    let window = &HANN_WINDOW;
    let mut spec = Array3::zeros((2, freq_bins, n_frames));
    spec.axis_iter_mut(Axis(2))
        .into_par_iter()
        .enumerate()
        .for_each(|(frame_idx, mut frame_view)| {
            let start = frame_idx * *HOP_SIZE;
            let slice_end = (start + FFT_SIZE).min(signal.len());
            let slice_len = slice_end - start;
            REAL_BUF.with(|cell| {
                let mut real_input = cell.borrow_mut();
                for (i, (&s, &w)) in signal[start..slice_end].iter().zip(window.iter()).enumerate() {
                    real_input[i] = s * w;
                }
                for i in slice_len..FFT_SIZE {
                    real_input[i] = 0.0;
                }
                RE_BUF.with(|rb| {
                    IM_BUF.with(|ib| {
                        let (mut re, mut im) = (rb.borrow_mut(), ib.borrow_mut());
                        r2c_fft_f32_with_planner(&real_input[..], re.as_mut_slice(), im.as_mut_slice(), planner);
                        let (mut spec_re, mut spec_im) = frame_view.multi_slice_mut((s![0, ..], s![1, ..]));
                        for (j, (&rv, &iv)) in re.iter().zip(im.iter()).enumerate() {
                            spec_re[j] = rv;
                            spec_im[j] = iv;
                        }
                    });
                });
            });
        });
    spec
}
pub fn istft_core(spec: &Array3<f32>, orig_len: usize) -> Vec<f32> {
    let freq_bins = FFT_SIZE / 2 + 1;
    let n_frames = spec.shape()[2];
    let planner = &*FFT_PLANNER;
    let mut output = vec![0.0; (n_frames - 1) * *HOP_SIZE + FFT_SIZE];
    let mut weight = vec![0.0; output.len()];
    let mut real_buf = vec![0.0; FFT_SIZE];
    let mut re_buf = vec![0.0; freq_bins];
    let mut im_buf = vec![0.0; freq_bins];
    let hann_fft = &*HANN_FFT_SCALE;
    let hann_sq = &*HANN_SQ;
    for frame_idx in 0..n_frames {
        for j in 0..freq_bins {
            re_buf[j] = spec[[0, j, frame_idx]];
            im_buf[j] = spec[[1, j, frame_idx]];
        }
        c2r_fft_f32_with_planner(&re_buf, &im_buf, &mut real_buf, planner);
        let start = frame_idx * *HOP_SIZE;
        let end = (start + FFT_SIZE).min(output.len());
        for i in 0..end - start {
            let pos = start + i;
            output[pos] += real_buf[i] * hann_fft[i];
            weight[pos] += hann_sq[i];
        }
    }
    for i in 0..orig_len.min(output.len()) {
        if weight[i] > 1e-10 {
            output[i] /= weight[i];
        }
    }
    output.truncate(orig_len);
    output
}