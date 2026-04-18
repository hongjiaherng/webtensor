use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

/// Strided relu: `max(0, a)`.
///
/// meta layout (19 × u32):
///   [0]      total elements
///   [1]      rank
///   [2..9]   shape[0..7]
///   [10..17] strides[0..7]
///   [18]     offset
#[wasm_bindgen]
pub fn relu_strided(a_ptr: *const f32, out_ptr: *mut f32, meta_ptr: *const u32) {
    unsafe {
        let meta = slice::from_raw_parts(meta_ptr, 19);
        let total = meta[0] as usize;
        let rank = meta[1] as usize;
        let shape = &meta[2..2 + rank];
        let strides = &meta[10..10 + rank];
        let offset = meta[18];
        let out = slice::from_raw_parts_mut(out_ptr, total);

        for i in 0..total {
            let ai = strided_idx(shape, strides, offset, i as u32);
            let v = *a_ptr.add(ai);
            out[i] = if v > 0.0 { v } else { 0.0 };
        }
    }
}
