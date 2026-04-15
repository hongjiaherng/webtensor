use std::slice;
use wasm_bindgen::prelude::*;

pub fn matmul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            let mut sum: f32 = 0.0;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            out[row * n + col] = sum;
        }
    }
}

#[wasm_bindgen]
pub fn matmul_raw(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    // SAFETY: pointers come from the WASM linear memory allocator (alloc_f32)
    // and are guaranteed to be valid, non-null, and correctly sized by the caller.
    unsafe {
        let a = slice::from_raw_parts(a_ptr, m * k);
        let b = slice::from_raw_parts(b_ptr, k * n);
        let out = slice::from_raw_parts_mut(out_ptr, m * n);
        matmul(a, b, out, m, k, n);
    }
}
