//! Element-wise comparison kernels: eq / ne / lt / le / gt / ge / isclose.
//!
//! Inputs share one arithmetic dtype (f32 or i32); output is u8 (bool 0/1).
//! They accept the same 28-u32 binary meta layout used by add/sub/mul/div.

use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

/// Generate a strided broadcast comparison: apply `$pred(a, b) -> bool`, write
/// the result to an u8 output as 0 / 1.
macro_rules! compare_kernel {
    ($name:ident, $scalar:ty, |$a:ident, $b:ident| $pred:expr) => {
        #[wasm_bindgen]
        pub fn $name(
            a_ptr: *const $scalar,
            b_ptr: *const $scalar,
            out_ptr: *mut u8,
            meta_ptr: *const u32,
        ) {
            unsafe {
                let meta = slice::from_raw_parts(meta_ptr, 28);
                let total = meta[0] as usize;
                let rank = meta[1] as usize;
                let shape = &meta[2..2 + rank];
                let a_bcast = &meta[10..10 + rank];
                let a_off = meta[18];
                let b_bcast = &meta[19..19 + rank];
                let b_off = meta[27];
                let out = slice::from_raw_parts_mut(out_ptr, total);
                for i in 0..total {
                    let ai = strided_idx(shape, a_bcast, a_off, i as u32);
                    let bi = strided_idx(shape, b_bcast, b_off, i as u32);
                    let $a: $scalar = *a_ptr.add(ai);
                    let $b: $scalar = *b_ptr.add(bi);
                    out[i] = if $pred { 1 } else { 0 };
                }
            }
        }
    };
}

compare_kernel!(eq_f32_strided, f32, |a, b| a == b);
compare_kernel!(eq_i32_strided, i32, |a, b| a == b);
compare_kernel!(ne_f32_strided, f32, |a, b| a != b);
compare_kernel!(ne_i32_strided, i32, |a, b| a != b);
compare_kernel!(lt_f32_strided, f32, |a, b| a < b);
compare_kernel!(lt_i32_strided, i32, |a, b| a < b);
compare_kernel!(le_f32_strided, f32, |a, b| a <= b);
compare_kernel!(le_i32_strided, i32, |a, b| a <= b);
compare_kernel!(gt_f32_strided, f32, |a, b| a > b);
compare_kernel!(gt_i32_strided, i32, |a, b| a > b);
compare_kernel!(ge_f32_strided, f32, |a, b| a >= b);
compare_kernel!(ge_i32_strided, i32, |a, b| a >= b);

/// `|a - b| <= atol + rtol * |b|`. Float-only. Tolerances and `equal_nan` are
/// passed as direct function arguments rather than meta fields.
#[wasm_bindgen]
pub fn isclose_f32_strided(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut u8,
    meta_ptr: *const u32,
    rtol: f32,
    atol: f32,
    equal_nan: u32,
) {
    unsafe {
        let meta = slice::from_raw_parts(meta_ptr, 28);
        let total = meta[0] as usize;
        let rank = meta[1] as usize;
        let shape = &meta[2..2 + rank];
        let a_bcast = &meta[10..10 + rank];
        let a_off = meta[18];
        let b_bcast = &meta[19..19 + rank];
        let b_off = meta[27];
        let out = slice::from_raw_parts_mut(out_ptr, total);
        let eq_nan = equal_nan != 0;

        for i in 0..total {
            let ai = strided_idx(shape, a_bcast, a_off, i as u32);
            let bi = strided_idx(shape, b_bcast, b_off, i as u32);
            let a = *a_ptr.add(ai);
            let b = *b_ptr.add(bi);

            let close = if a.is_nan() || b.is_nan() {
                eq_nan && a.is_nan() && b.is_nan()
            } else if a.is_infinite() || b.is_infinite() {
                // Infinities match only on exact sign; tolerance math would
                // incorrectly report |inf - (-inf)| <= inf as close.
                a == b
            } else {
                (a - b).abs() <= atol + rtol * b.abs()
            };
            out[i] = if close { 1 } else { 0 };
        }
    }
}
