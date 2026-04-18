//! Concat kernel — one Rust call scatters a single input tensor into its slice
//! of the output along `axis`. The TS wrapper invokes it once per input,
//! advancing `axis_start` between calls. Meta layout: see
//! `ops::CONCAT_META_WORDS`.

use crate::ops::{
    CONCAT_AXIS_OFF, CONCAT_AXIS_START_OFF, CONCAT_IN_OFFSET_OFF, CONCAT_IN_SHAPE_OFF,
    CONCAT_IN_STRIDES_OFF, CONCAT_META_WORDS, CONCAT_OUT_SHAPE_OFF, MAX_RANK,
};
use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

macro_rules! concat_kernel {
    ($name:ident, $scalar:ty) => {
        #[wasm_bindgen]
        pub fn $name(a_ptr: *const $scalar, out_ptr: *mut $scalar, meta_ptr: *const u32) {
            unsafe {
                let meta = slice::from_raw_parts(meta_ptr, CONCAT_META_WORDS);
                let total = meta[0] as usize;
                let rank = meta[1] as usize;
                let in_shape = &meta[CONCAT_IN_SHAPE_OFF..CONCAT_IN_SHAPE_OFF + rank];
                let in_strides = &meta[CONCAT_IN_STRIDES_OFF..CONCAT_IN_STRIDES_OFF + rank];
                let in_offset = meta[CONCAT_IN_OFFSET_OFF];
                let out_shape = &meta[CONCAT_OUT_SHAPE_OFF..CONCAT_OUT_SHAPE_OFF + rank];
                let axis = meta[CONCAT_AXIS_OFF] as usize;
                let axis_start = meta[CONCAT_AXIS_START_OFF];

                // Contiguous row-major strides for the output, computed here so
                // we don't have to pack them into meta.
                let mut out_strides = [0u32; MAX_RANK];
                let mut acc: u32 = 1;
                for d in (0..rank).rev() {
                    out_strides[d] = acc;
                    acc *= out_shape[d];
                }

                for i in 0..total {
                    let a_idx = strided_idx(in_shape, in_strides, in_offset, i as u32);
                    let v = *a_ptr.add(a_idx);

                    // Unravel `i` against input shape, shift axis coord, re-ravel
                    // against the (contiguous) output strides.
                    let mut rem = i as u32;
                    let mut out_flat: u32 = 0;
                    for d in (0..rank).rev() {
                        let dim = in_shape[d];
                        let coord = rem % dim;
                        rem /= dim;
                        let out_coord = if d == axis { coord + axis_start } else { coord };
                        out_flat += out_coord * out_strides[d];
                    }
                    *out_ptr.add(out_flat as usize) = v;
                }
            }
        }
    };
}

concat_kernel!(concat_f32_strided, f32);
concat_kernel!(concat_i32_strided, i32);
concat_kernel!(concat_u8_strided, u8);
