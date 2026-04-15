use wasm_bindgen::prelude::*;
use std::slice;

#[wasm_bindgen]
pub fn relu(a: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
    }
}

#[wasm_bindgen]
pub unsafe fn relu_raw(a_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    let a = slice::from_raw_parts(a_ptr, len);
    let out = slice::from_raw_parts_mut(out_ptr, len);
    relu(a, out);
}

#[wasm_bindgen]
pub unsafe fn relu_grad_raw(grad_ptr: *const f32, a_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    let grad = slice::from_raw_parts(grad_ptr, len);
    let a = slice::from_raw_parts(a_ptr, len);
    let out = slice::from_raw_parts_mut(out_ptr, len);
    for i in 0..len {
        out[i] = if a[i] > 0.0 { grad[i] } else { 0.0 };
    }
}
