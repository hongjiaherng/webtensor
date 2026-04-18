use crate::utils::strided_idx;
use std::slice;
use wasm_bindgen::prelude::*;

/// Strided element-wise divide. Same meta layout as add (28 × u32).

#[wasm_bindgen]
pub fn div_f32_strided(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    meta_ptr: *const u32,
) {
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

#[wasm_bindgen]
pub fn div_i32_strided(
    a_ptr: *const i32,
    b_ptr: *const i32,
    out_ptr: *mut i32,
    meta_ptr: *const u32,
) {
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
            let b_val = *b_ptr.add(bi);
            // i32 div by zero is a trap in Rust; JS tensor division also silently
            // yields 0 for int div by 0 — mirror that by checking.
            out[i] = if b_val == 0 { 0 } else { (*a_ptr.add(ai)) / b_val };
        }
    }
}
