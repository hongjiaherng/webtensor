use std::slice;
use wasm_bindgen::prelude::*;

/// Element-wise divide with suffix broadcasting: a[i % len_a] / b[i % len_b].
pub fn div(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = a[i % a.len()] / b[i % b.len()];
    }
}

/// Pointer-based entry point called from JavaScript.
#[wasm_bindgen]
pub fn div_raw(
    a_ptr: *const f32, b_ptr: *const f32, out_ptr: *mut f32,
    len_a: usize, len_b: usize, len_out: usize,
) {
    // SAFETY: pointers come from the WASM linear memory allocator (alloc_f32)
    // and are guaranteed to be valid, non-null, and correctly sized by the caller.
    unsafe {
        let a = slice::from_raw_parts(a_ptr, len_a);
        let b = slice::from_raw_parts(b_ptr, len_b);
        let out = slice::from_raw_parts_mut(out_ptr, len_out);
        div(a, b, out);
    }
}
