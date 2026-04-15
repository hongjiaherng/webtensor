use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

/// Strided element-wise divide.  Same meta layout as add_strided (28 × u32).
#[wasm_bindgen]
pub fn div_strided(a_ptr: *const f32, b_ptr: *const f32, out_ptr: *mut f32, meta_ptr: *const u32) {
    unsafe {
        let meta = slice::from_raw_parts(meta_ptr, 28);
        let total = meta[0] as usize;
        let rank = meta[1] as usize;
        let shape = &meta[2..2 + rank];
        let a_bcast = &meta[10..10 + rank];
        let a_off = meta[18];
        let b_bcast = &meta[19..19 + rank];
        let b_off = meta[27];
        let out = slice::from_raw_parts_mut(out_ptr, total);

        for i in 0..total {
            let ai = strided_idx(shape, a_bcast, a_off, i as u32);
            let bi = strided_idx(shape, b_bcast, b_off, i as u32);
            out[i] = *a_ptr.add(ai) / *b_ptr.add(bi);
        }
    }
}
