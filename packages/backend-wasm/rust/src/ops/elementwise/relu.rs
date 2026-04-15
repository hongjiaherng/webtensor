use std::slice;
use wasm_bindgen::prelude::*;

pub fn relu(a: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
    }
}

/// Backward pass: passes gradient where the forward input was positive, zeros elsewhere.
pub fn relu_grad(grad: &[f32], a: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = if a[i] > 0.0 { grad[i] } else { 0.0 };
    }
}

#[wasm_bindgen]
pub fn relu_raw(a_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    // SAFETY: pointers come from the WASM linear memory allocator (alloc_f32)
    // and are guaranteed to be valid, non-null, and correctly sized by the caller.
    unsafe {
        let a = slice::from_raw_parts(a_ptr, len);
        let out = slice::from_raw_parts_mut(out_ptr, len);
        relu(a, out);
    }
}

#[wasm_bindgen]
pub fn relu_grad_raw(grad_ptr: *const f32, a_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    // SAFETY: pointers come from the WASM linear memory allocator (alloc_f32)
    // and are guaranteed to be valid, non-null, and correctly sized by the caller.
    unsafe {
        let grad = slice::from_raw_parts(grad_ptr, len);
        let a = slice::from_raw_parts(a_ptr, len);
        let out = slice::from_raw_parts_mut(out_ptr, len);
        relu_grad(grad, a, out);
    }
}
