use num_traits::AsPrimitive;

use crate::types::Num;

// ------------ AVERAGE

// TODO: future work -> this can be optimized by using SIMD instructions (similar to the argminmax crate)
// TODO: this implementation can overfow (but numpy does the same)

// This trait implements the average function for all types that this crate
// supports. It is used in the lttb algorithm.
// We intend to use the same implementation for all types as is used in the
// numpy (Python) library (- which uses add reduce):
//  - f64 & f32: use the data type to calculate the average
//  - f16: cast to f32 and calculate the average
//  - signed & unsigned integers: cast to f64 and calculate the average
// Note: the only difference with the numpy implementation is that this
// implementation always returns a f64, while numpy returns f32 for f32 and f16
// (however the calculation is done in f32 - only the result is casted to f64).
// See more details: https://github.com/numpy/numpy/blob/8cec82012694571156e8d7696307c848a7603b4e/numpy/core/_methods.py#L164

pub trait Average {
    fn average(&self) -> f64;
    fn average_optimized(&self) -> f64;
}

impl<T> Average for [T]
where
    T: Num + AsPrimitive<f64>,
{
    fn average(&self) -> f64 {
        self.iter().fold(0f64, |acc, &x| acc + x.as_()) as f64 / self.len() as f64
    }

    fn average_optimized(&self) -> f64 {
        // Dispatch to SIMD implementation based on runtime CPU detection
        average_simd_dispatch(self)
    }
}

// SIMD-optimized averaging with runtime CPU detection
#[inline]
fn average_simd_dispatch<T: Num + AsPrimitive<f64>>(data: &[T]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { average_f64_avx2(data) }
        } else {
            average_f64_scalar(data)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        average_f64_scalar(data)
    }
}

// Scalar fallback (same as original)
#[inline]
fn average_f64_scalar<T: Num + AsPrimitive<f64>>(data: &[T]) -> f64 {
    data.iter().fold(0f64, |acc, &x| acc + x.as_()) / data.len() as f64
}

// AVX2 SIMD implementation for f64 accumulation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn average_f64_avx2<T: Num + AsPrimitive<f64>>(data: &[T]) -> f64 {
    use std::arch::x86_64::*;

    let len = data.len();
    if len == 0 {
        return 0.0;
    }

    // Convert to f64 and use AVX2
    // AVX2 processes 4 f64s at a time (256-bit / 64-bit = 4)
    let mut sum = _mm256_setzero_pd();
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= len {
        let a = data[i].as_();
        let b = data[i + 1].as_();
        let c = data[i + 2].as_();
        let d = data[i + 3].as_();

        let vals = _mm256_set_pd(d, c, b, a);
        sum = _mm256_add_pd(sum, vals);
        i += 4;
    }

    // Horizontal sum of the 4 f64 lanes
    let sum_array: [f64; 4] = std::mem::transmute(sum);
    let mut total = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle remaining elements
    while i < len {
        total += data[i].as_();
        i += 1;
    }

    total / len as f64
}
