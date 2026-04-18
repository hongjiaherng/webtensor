use crate::kernels::{UNARY_META_WORDS, UNARY_OFFSET_OFF, UNARY_SHAPE_OFF, UNARY_STRIDES_OFF};
use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn log_strided(a_ptr: *const f32, out_ptr: *mut f32, meta_ptr: *const u32) {
    unsafe {
        let meta = slice::from_raw_parts(meta_ptr, UNARY_META_WORDS);
        let total = meta[0] as usize;
        let rank = meta[1] as usize;
        let shape = &meta[UNARY_SHAPE_OFF..UNARY_SHAPE_OFF + rank];
        let strides = &meta[UNARY_STRIDES_OFF..UNARY_STRIDES_OFF + rank];
        let offset = meta[UNARY_OFFSET_OFF];
        let out = slice::from_raw_parts_mut(out_ptr, total);

        for i in 0..total {
            let ai = strided_idx(shape, strides, offset, i as u32);
            out[i] = (*a_ptr.add(ai)).ln();
        }
    }
}
