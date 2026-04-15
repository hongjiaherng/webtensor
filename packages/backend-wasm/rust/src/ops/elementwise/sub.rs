use wasm_bindgen::prelude::*;
use std::slice;

#[wasm_bindgen]
pub fn sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = a[i] - b[i];
    }
}

#[wasm_bindgen]
pub unsafe fn sub_raw(a_ptr: *const f32, b_ptr: *const f32, out_ptr: *mut f32, len_a: usize, len_b: usize, len_out: usize) {
    let a = slice::from_raw_parts(a_ptr, len_a);
    let b = slice::from_raw_parts(b_ptr, len_b);
    let out = slice::from_raw_parts_mut(out_ptr, len_out);
    for i in 0..len_out {
        out[i] = a[i % len_a] - b[i % len_b];
    }
}
