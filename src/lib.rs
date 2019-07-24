/// Initial implementations inspired by
/// https://medium.com/@Razican/learning-simd-with-rust-by-finding-planets-b85ccfb724c3.
pub mod mars;

use std::{f64, mem};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

/// Implementation taken from
/// https://medium.com/@Razican/learning-simd-with-rust-by-finding-planets-b85ccfb724c3.
pub fn calculate_var(t: f64, var: &[(f64, f64, f64)]) -> f64 {
    var.iter()
        .fold(0_f64, |term, &(a, b, c)| term + a * (b + c * t).cos())
}

/// Implementation taken from
/// https://medium.com/@Razican/learning-simd-with-rust-by-finding-planets-b85ccfb724c3.
#[inline]
pub fn calculate_var_inline(t: f64, var: &[(f64, f64, f64)]) -> f64 {
    var.iter()
        .fold(0_f64, |term, &(a, b, c)| term + a * (b + c * t).cos())
}

/// Implementation taken from
/// https://medium.com/@Razican/learning-simd-with-rust-by-finding-planets-b85ccfb724c3.
pub unsafe fn vector_term(
    (a1, b1, c1): (f64, f64, f64),
    (a2, b2, c2): (f64, f64, f64),
    (a3, b3, c3): (f64, f64, f64),
    (a4, b4, c4): (f64, f64, f64),
    t: f64,
) -> (f64, f64, f64, f64) {
    let a = _mm256_set_pd(a1, a2, a3, a4);
    let b = _mm256_set_pd(b1, b2, b3, b4);
    let c = _mm256_set_pd(c1, c2, c3, c4);
    let t = _mm256_set1_pd(t);
    // Safe because both values are created properly and checked.
    let ct = _mm256_mul_pd(c, t);
    // Safe because both values are created properly and checked.
    let bct = _mm256_add_pd(b, ct);
    let bct_unpacked: (f64, f64, f64, f64) = mem::transmute(bct);
    let bct = _mm256_set_pd(
        bct_unpacked.3.cos(),
        bct_unpacked.2.cos(),
        bct_unpacked.1.cos(),
        bct_unpacked.0.cos(),
    );
    let term = _mm256_mul_pd(a, bct);
    mem::transmute(term)
}

/// Implementation taken from
/// https://medium.com/@Razican/learning-simd-with-rust-by-finding-planets-b85ccfb724c3.
#[target_feature(enable = "avx")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub unsafe fn calculate_var_avx_inner(t: f64, var: &[(f64, f64, f64)]) -> f64 {
    var.chunks(4)
        .map(|vec| match vec {
            &[(a1, b1, c1), (a2, b2, c2), (a3, b3, c3), (a4, b4, c4)] => {
                // The result is little endian in x86/x86_64.
                let (term4, term3, term2, term1) =
                    vector_term((a1, b1, c1), (a2, b2, c2), (a3, b3, c3), (a4, b4, c4), t);
                term1 + term2 + term3 + term4
            }
            &[(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)] => {
                let (_term4, term3, term2, term1) = vector_term(
                    (a1, b1, c1),
                    (a2, b2, c2),
                    (a3, b3, c3),
                    (f64::NAN, f64::NAN, f64::NAN),
                    t,
                );
                term1 + term2 + term3
            }
            &[(a1, b1, c1), (a2, b2, c2)] => a1 * (b1 + c1 * t).cos() + a2 * (b2 + c2 * t).cos(),
            &[(a, b, c)] => a * (b + c * t).cos(),
            _ => unimplemented!(),
        })
        .sum()
}

/// Implementation taken from
/// https://medium.com/@Razican/learning-simd-with-rust-by-finding-planets-b85ccfb724c3.
#[inline]
#[allow(unsafe_code)]
pub fn calculate_var_avx(t: f64, var: &[(f64, f64, f64)]) -> f64 {
    if is_x86_feature_detected!("avx") {
        // Safe because we already checked that we have the AVX instruction
        // set.
        unsafe { calculate_var_avx_inner(t, var) }
    } else {
        var.iter()
            .fold(0_f64, |term, &(a, b, c)| term + a * (b + c * t).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_impls_are_consistent() {
        // TODO machine precision (epsilon)?
        let thresh = 1.0e-14;
        assert!(
            (calculate_var(1989.0, &mars::A0) - calculate_var_avx(1989.0, &mars::A0)).abs()
                < thresh
        );
    }
}
