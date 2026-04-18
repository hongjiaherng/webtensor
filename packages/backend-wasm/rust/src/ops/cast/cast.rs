//! Dtype conversion kernels. One `#[wasm_bindgen]` function per (from, to) pair.
//! All cast kernels are strided and accept the shared unary meta layout
//! (see `ops::UNARY_META_WORDS`). Output is contiguous.

use crate::ops::{UNARY_META_WORDS, UNARY_OFFSET_OFF, UNARY_SHAPE_OFF, UNARY_STRIDES_OFF};
use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

/// Generate a strided cast kernel. `$body` receives a local `v: $from` binding
/// and must evaluate to `$to`.
macro_rules! cast_kernel {
    ($name:ident, $from:ty, $to:ty, |$v:ident| $body:expr) => {
        #[wasm_bindgen]
        pub fn $name(a_ptr: *const $from, out_ptr: *mut $to, meta_ptr: *const u32) {
            unsafe {
                let meta = slice::from_raw_parts(meta_ptr, UNARY_META_WORDS);
                let total = meta[0] as usize;
                let rank = meta[1] as usize;
                let shape = &meta[UNARY_SHAPE_OFF..UNARY_SHAPE_OFF + rank];
                let strides = &meta[UNARY_STRIDES_OFF..UNARY_STRIDES_OFF + rank];
                let offset = meta[UNARY_OFFSET_OFF];
                let out = slice::from_raw_parts_mut(out_ptr, total);
                for i in 0..total {
                    let idx = strided_idx(shape, strides, offset, i as u32);
                    let $v: $from = *a_ptr.add(idx);
                    out[i] = $body;
                }
            }
        }
    };
}

// ------------- numeric casts (truncate / widen) -------------

cast_kernel!(cast_f32_i32_strided, f32, i32, |v| v as i32);
cast_kernel!(cast_i32_f32_strided, i32, f32, |v| v as f32);

// ------------- *-to-bool: canonicalize to 0 / 1 -------------

cast_kernel!(cast_f32_u8_strided, f32, u8, |v| if v != 0.0 { 1 } else { 0 });
cast_kernel!(cast_i32_u8_strided, i32, u8, |v| if v != 0 { 1 } else { 0 });

// ------------- bool-to-numeric: already 0 / 1, just widen -------------

cast_kernel!(cast_u8_f32_strided, u8, f32, |v| v as f32);
cast_kernel!(cast_u8_i32_strided, u8, i32, |v| v as i32);

// ------------- same-dtype (pure copies with strided read) -------------

cast_kernel!(cast_f32_f32_strided, f32, f32, |v| v);
cast_kernel!(cast_i32_i32_strided, i32, i32, |v| v);
cast_kernel!(cast_u8_u8_strided, u8, u8, |v| v);
