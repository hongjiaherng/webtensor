use crate::ops::{MAX_RANK, SOFTMAX_META_WORDS, SOFTMAX_SHAPE_OFF, SOFTMAX_STRIDES_OFF};
use std::slice;
use wasm_bindgen::prelude::*;

/// Strided softmax for float32. Output is contiguous with same shape as input.
/// Meta layout: see `ops::SOFTMAX_META_WORDS`.
#[wasm_bindgen]
pub fn softmax_f32_strided(a_ptr: *const f32, out_ptr: *mut f32, meta_ptr: *const u32) {
    unsafe {
        let meta = slice::from_raw_parts(meta_ptr, SOFTMAX_META_WORDS);
        let rank = meta[0] as usize;
        let axis = meta[1] as usize;
        let offset = meta[2] as usize;
        let shape = &meta[SOFTMAX_SHAPE_OFF..SOFTMAX_SHAPE_OFF + rank];
        let strides = &meta[SOFTMAX_STRIDES_OFF..SOFTMAX_STRIDES_OFF + rank];
        let axis_len = shape[axis] as usize;

        // Compute contiguous out_strides from shape.
        let mut out_strides = [1usize; MAX_RANK];
        for i in (0..rank.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * shape[i + 1] as usize;
        }
        let out_axis_stride = out_strides[axis];
        let in_axis_stride = strides[axis] as usize;

        let mut total: usize = 1;
        for i in 0..rank {
            total *= shape[i] as usize;
        }
        let slice_count = total / axis_len;

        let out = slice::from_raw_parts_mut(out_ptr, total);

        let mut coord = [0u32; MAX_RANK];
        for o in 0..slice_count {
            let mut rem = o;
            for d in (0..rank).rev() {
                if d == axis {
                    continue;
                }
                let dim = shape[d] as usize;
                coord[d] = (rem % dim) as u32;
                rem /= dim;
            }
            coord[axis] = 0;

            let mut in_base = offset;
            let mut out_base: usize = 0;
            for d in 0..rank {
                in_base += (coord[d] as usize) * strides[d] as usize;
                out_base += (coord[d] as usize) * out_strides[d];
            }

            let mut max_v = f32::NEG_INFINITY;
            for k in 0..axis_len {
                let v = *a_ptr.add(in_base + k * in_axis_stride);
                if v > max_v {
                    max_v = v;
                }
            }

            let mut sum: f32 = 0.0;
            for k in 0..axis_len {
                let e = (*a_ptr.add(in_base + k * in_axis_stride) - max_v).exp();
                out[out_base + k * out_axis_stride] = e;
                sum += e;
            }

            let inv_sum = if sum == 0.0 { 0.0 } else { 1.0_f32 / sum };
            for k in 0..axis_len {
                out[out_base + k * out_axis_stride] *= inv_sum;
            }
        }
    }
}
