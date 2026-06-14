//! Inverse Modified Discrete Cosine Transform (IMDCT).
//!
//! Shared between AAC (windows 256/2048) and MP3 Layer III (windows
//! 36 / 12). V1 ships the direct-form definition — O(N²) per block,
//! which is fine for the lengths involved in tests and the
//! decoder-correctness pass. A radix-2 FFT-based IMDCT (the path
//! production codecs take) lands as a follow-up; the direct form
//! is the reference the FFT path validates against.
//!
//! Definition (matching the IMDCT used in ISO/IEC 11172 + 14496-3):
//!   x[n] = (2/N) · Σ_{k=0..N/2-1} X[k] · cos( π/(2N) · (2n + 1 + N/2) · (2k + 1) )

use std::f64::consts::PI;

/// Run an N-point IMDCT. Input has N/2 coefficients; output is N
/// time-domain samples. `n` must be even and a small power-of-two
/// in practice (the AAC choices: 2048 or 256; MP3: 36 or 12).
pub fn imdct(input: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(input.len() * 2, n);
    let mut out = vec![0.0; n];
    let scale = 2.0 / (n as f64);
    let n_half = (n / 2) as f64;
    for (sample_n, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for k in 0..input.len() {
            let arg = PI / (2.0 * n as f64)
                * (2.0 * sample_n as f64 + 1.0 + n_half)
                * (2.0 * k as f64 + 1.0);
            acc += input[k] * arg.cos();
        }
        *slot = scale * acc;
    }
    out
}

/// Sine window — both AAC and MP3 default to this.
pub fn sine_window(n: usize) -> Vec<f64> {
    let mut w = vec![0.0; n];
    for i in 0..n {
        w[i] = (PI / n as f64 * (i as f64 + 0.5)).sin();
    }
    w
}

/// Overlap-add the second half of the previous block with the first
/// half of the current block. Returns N/2 output samples.
pub fn overlap_add(prev_tail: &[f64], cur_head: &[f64]) -> Vec<f64> {
    assert_eq!(prev_tail.len(), cur_head.len());
    prev_tail.iter().zip(cur_head).map(|(a, b)| a + b).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imdct_dc_only_is_constant_modulated_by_cosine() {
        // Pure DC coefficient: input = [1, 0, 0, ...]. Output is a
        // weighted cosine sweep — bounded by ±2/N and not all zero.
        let n = 8;
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let out = imdct(&input, n);
        let nonzero: f64 = out.iter().map(|v| v.abs()).sum();
        assert!(nonzero > 0.0);
        for &v in &out {
            assert!(v.abs() <= 2.0 / n as f64 + 1e-9, "got {v}");
        }
    }

    #[test]
    fn sine_window_sums_to_n_over_2() {
        // ∑ sin²(πi/N · (i+0.5)) = N/2 for the standard sine window.
        let n = 16;
        let w = sine_window(n);
        let s: f64 = w.iter().map(|x| x * x).sum();
        assert!((s - (n as f64 / 2.0)).abs() < 1e-9);
    }

    #[test]
    fn overlap_add_pairs_corresponding_samples() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let r = overlap_add(&a, &b);
        assert_eq!(r, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn imdct_zero_input_yields_zero_output() {
        let out = imdct(&vec![0.0; 8], 16);
        for v in out {
            assert!(v.abs() < 1e-12);
        }
    }
}
