use crate::{consts::{FFT_SIZE, HOP_SIZE}, utils::hann_window::HANN_WINDOW};
use ndarray::{Array3, ArrayView1, Axis, parallel::prelude::*, s};
use once_cell::sync::Lazy;
use phastft::{c2r_fft_f32_with_planner, planner::PlannerR2c32, r2c_fft_f32_with_planner};
static FFT_PLANNER: Lazy<PlannerR2c32> = Lazy::new(|| PlannerR2c32::new(FFT_SIZE));
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
                RE_BUF.with(|re_cell| {
                    IM_BUF.with(|im_cell| {
                        let mut spec_re = re_cell.borrow_mut();
                        let mut spec_im = im_cell.borrow_mut();
                        r2c_fft_f32_with_planner(&real_input[..], &mut spec_re[..], &mut spec_im[..], planner);
                        frame_view.row_mut(0).assign(&ArrayView1::from(&*spec_re));
                        frame_view.row_mut(1).assign(&ArrayView1::from(&*spec_im));
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
    for frame_idx in 0..n_frames {
        let re_slice = spec.slice(s![0, .., frame_idx]);
        let im_slice = spec.slice(s![1, .., frame_idx]);
        re_buf.copy_from_slice(re_slice.as_slice().unwrap());
        im_buf.copy_from_slice(im_slice.as_slice().unwrap());
        c2r_fft_f32_with_planner(&re_buf, &im_buf, &mut real_buf, planner);
        let start = frame_idx * *HOP_SIZE;
        for i in 0..FFT_SIZE {
            let val = real_buf[i] / FFT_SIZE as f32 * HANN_WINDOW[i];
            let pos = start + i;
            if pos < output.len() {
                output[pos] += val;
                weight[pos] += HANN_WINDOW[i] * HANN_WINDOW[i];
            }
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