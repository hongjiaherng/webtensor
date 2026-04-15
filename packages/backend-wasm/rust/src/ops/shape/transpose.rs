use std::slice;
use wasm_bindgen::prelude::*;

pub fn transpose(a: &[f32], out: &mut [f32], m: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            out[col * m + row] = a[row * n + col];
        }
    }
}

#[wasm_bindgen]
pub fn transpose_raw(a_ptr: *const f32, out_ptr: *mut f32, m: usize, n: usize) {
    // SAFETY: pointers come from the WASM linear memory allocator (alloc_f32)
    // and are guaranteed to be valid, non-null, and correctly sized by the caller.
    unsafe {
        let a = slice::from_raw_parts(a_ptr, m * n);
        let out = slice::from_raw_parts_mut(out_ptr, m * n);
        transpose(a, out, m, n);
    }
}
