//! Concat kernel — one Rust call scatters a single input tensor into its slice
//! of the output along `axis`. The TS wrapper invokes it once per input,
//! advancing `axis_start` between calls. This keeps the kernel signature fixed
//! (one input, one output) despite Concat being variadic at the IR level.
//!
//! meta layout (29 × u32):
//!   [0]       total (input elements)
//!   [1]       rank
//!   [2..9]    in_shape[0..7]
//!   [10..17]  in_strides[0..7]
//!   [18]      in_offset
//!   [19..26]  out_shape[0..7]
//!   [27]      axis
//!   [28]      axis_start

use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

macro_rules! concat_kernel {
    ($name:ident, $scalar:ty) => {
        #[wasm_bindgen]
        pub fn $name(a_ptr: *const $scalar, out_ptr: *mut $scalar, meta_ptr: *const u32) {
            unsafe {
                let meta = slice::from_raw_parts(meta_ptr, 29);
                let total = meta[0] as usize;
                let rank = meta[1] as usize;
                let in_shape = &meta[2..2 + rank];
                let in_strides = &meta[10..10 + rank];
                let in_offset = meta[18];
                let out_shape = &meta[19..19 + rank];
                let axis = meta[27] as usize;
                let axis_start = meta[28];

                // Contiguous row-major strides for the output, computed here so
                // we don't have to pack them into meta.
                let mut out_strides = [0u32; 8];
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
