//! Constant-value Pad (ONNX `Pad`, mode="constant"). Fills the output with
//! `value`, then copies the input into the region starting at `pads_before`.
//!
//! meta layout (36 × u32):
//!   [0]       src_total
//!   [1]       rank
//!   [2..9]    src_shape[0..7]
//!   [10..17]  src_strides[0..7]
//!   [18]      src_offset
//!   [19..26]  out_shape[0..7]
//!   [27..34]  pads_before[0..7]
//!   [35]      padding

use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

macro_rules! pad_kernel {
    ($name:ident, $scalar:ty) => {
        #[wasm_bindgen]
        pub fn $name(
            src_ptr: *const $scalar,
            out_ptr: *mut $scalar,
            meta_ptr: *const u32,
            value: $scalar,
        ) {
            unsafe {
                let meta = slice::from_raw_parts(meta_ptr, 36);
                let src_total = meta[0] as usize;
                let rank = meta[1] as usize;
                let src_shape = &meta[2..2 + rank];
                let src_strides = &meta[10..10 + rank];
                let src_offset = meta[18];
                let out_shape = &meta[19..19 + rank];
                let pads_before = &meta[27..27 + rank];

                let mut out_strides = [0u32; 8];
                let mut out_total: usize = 1;
                let mut acc: u32 = 1;
                for d in (0..rank).rev() {
                    out_strides[d] = acc;
                    acc *= out_shape[d];
                    out_total *= out_shape[d] as usize;
                }

                let out = slice::from_raw_parts_mut(out_ptr, out_total);
                for i in 0..out_total {
                    out[i] = value;
                }

                for i in 0..src_total {
                    let a_idx = strided_idx(src_shape, src_strides, src_offset, i as u32);
                    let v = *src_ptr.add(a_idx);

                    let mut rem = i as u32;
                    let mut out_flat: u32 = 0;
                    for d in (0..rank).rev() {
                        let dim = src_shape[d];
                        let coord = rem % dim;
                        rem /= dim;
                        out_flat += (coord + pads_before[d]) * out_strides[d];
                    }
                    *out_ptr.add(out_flat as usize) = v;
                }
            }
        }
    };
}

pad_kernel!(pad_f32_strided, f32);
pad_kernel!(pad_i32_strided, i32);
pad_kernel!(pad_u8_strided, u8);
