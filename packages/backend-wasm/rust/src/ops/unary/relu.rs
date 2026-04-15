use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

/// Strided relu.
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

/// Backward pass: passes gradient where the forward input was positive, zeros elsewhere.
/// Takes contiguous inputs (called from the autograd engine which always allocates fresh tensors).
#[wasm_bindgen]
pub fn relu_grad_raw(grad_ptr: *const f32, a_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    unsafe {
        let grad = slice::from_raw_parts(grad_ptr, len);
        let a = slice::from_raw_parts(a_ptr, len);
        let out = slice::from_raw_parts_mut(out_ptr, len);
        for i in 0..len {
            out[i] = if a[i] > 0.0 { grad[i] } else { 0.0 };
        }
    }
}
