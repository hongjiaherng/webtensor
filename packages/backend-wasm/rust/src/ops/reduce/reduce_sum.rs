use crate::ops::{
    MAX_RANK, REDUCE_AXES_OFF, REDUCE_META_WORDS, REDUCE_SHAPE_OFF, REDUCE_STRIDES_OFF,
};
use std::slice;
use wasm_bindgen::prelude::*;

/// Strided reduce-sum for float32. Meta layout: see `ops::REDUCE_META_WORDS`.
#[wasm_bindgen]
pub fn reduce_sum_f32_strided(a_ptr: *const f32, out_ptr: *mut f32, meta_ptr: *const u32) {
    unsafe {
        let meta = slice::from_raw_parts(meta_ptr, REDUCE_META_WORDS);
        let in_rank = meta[0] as usize;
        let reduce_rank = meta[1] as usize;
        let offset = meta[2] as usize;
        let in_shape = &meta[REDUCE_SHAPE_OFF..REDUCE_SHAPE_OFF + in_rank];
        let in_strides = &meta[REDUCE_STRIDES_OFF..REDUCE_STRIDES_OFF + in_rank];
        let axes = &meta[REDUCE_AXES_OFF..REDUCE_AXES_OFF + reduce_rank];

        // Mark which axes are reduced
        let mut is_reduce = [false; MAX_RANK];
        for i in 0..reduce_rank {
            is_reduce[axes[i] as usize] = true;
        }

        // Kept axes (in order)
        let mut kept_axes = [0usize; MAX_RANK];
        let mut kept_count = 0usize;
        let mut kept_total: usize = 1;
        for d in 0..in_rank {
            if !is_reduce[d] {
                kept_axes[kept_count] = d;
                kept_count += 1;
                kept_total *= in_shape[d] as usize;
            }
        }

        let mut reduce_total: usize = 1;
        for i in 0..reduce_rank {
            reduce_total *= in_shape[axes[i] as usize] as usize;
        }

        let out = slice::from_raw_parts_mut(out_ptr, kept_total);
        let mut coord = [0u32; MAX_RANK];

        for out_idx in 0..kept_total {
            let mut rem = out_idx;
            for i in (0..kept_count).rev() {
                let a = kept_axes[i];
                let d = in_shape[a] as usize;
                coord[a] = (rem % d) as u32;
                rem /= d;
            }

            let mut acc: f32 = 0.0;
            for r_idx in 0..reduce_total {
                let mut r_rem = r_idx;
                for i in (0..reduce_rank).rev() {
                    let a = axes[i] as usize;
                    let d = in_shape[a] as usize;
                    coord[a] = (r_rem % d) as u32;
                    r_rem /= d;
                }
                let mut off = offset;
                for d in 0..in_rank {
                    off += (coord[d] * in_strides[d]) as usize;
                }
                acc += *a_ptr.add(off);
            }
            out[out_idx] = acc;
        }
    }
}
