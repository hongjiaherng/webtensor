//! `|a - b| <= atol + rtol * |b|`. Float-only. Tolerances and `equal_nan` are
//! passed as direct function arguments rather than meta fields.

use crate::kernels::{
    BINARY_A_OFFSET_OFF, BINARY_A_STRIDES_OFF, BINARY_B_OFFSET_OFF, BINARY_B_STRIDES_OFF,
    BINARY_META_WORDS, BINARY_SHAPE_OFF,
};
use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

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
        let meta = slice::from_raw_parts(meta_ptr, BINARY_META_WORDS);
        let total = meta[0] as usize;
        let rank = meta[1] as usize;
        let shape = &meta[BINARY_SHAPE_OFF..BINARY_SHAPE_OFF + rank];
        let a_bcast = &meta[BINARY_A_STRIDES_OFF..BINARY_A_STRIDES_OFF + rank];
        let a_off = meta[BINARY_A_OFFSET_OFF];
        let b_bcast = &meta[BINARY_B_STRIDES_OFF..BINARY_B_STRIDES_OFF + rank];
        let b_off = meta[BINARY_B_OFFSET_OFF];
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
                a == b
            } else {
                (a - b).abs() <= atol + rtol * b.abs()
            };
            out[i] = if close { 1 } else { 0 };
        }
    }
}
