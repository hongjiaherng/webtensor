use wasm_bindgen::prelude::*;
use std::slice;

#[wasm_bindgen]
pub fn transpose(a: &[f32], out: &mut [f32], m: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            out[col * m + row] = a[row * n + col];
        }
    }
}

#[wasm_bindgen]
pub unsafe fn transpose_raw(a_ptr: *const f32, out_ptr: *mut f32, m: usize, n: usize) {
    let a = slice::from_raw_parts(a_ptr, m * n);
    let out = slice::from_raw_parts_mut(out_ptr, m * n);
    transpose(a, out, m, n);
}
