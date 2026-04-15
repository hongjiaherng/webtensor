use wasm_bindgen::prelude::*;
use std::slice;

#[wasm_bindgen]
pub fn sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = a[i] - b[i];
    }
}

#[wasm_bindgen]
pub unsafe fn sub_raw(a_ptr: *const f32, b_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    let a = slice::from_raw_parts(a_ptr, len);
    let b = slice::from_raw_parts(b_ptr, len);
    let out = slice::from_raw_parts_mut(out_ptr, len);
    sub(a, b, out);
}
