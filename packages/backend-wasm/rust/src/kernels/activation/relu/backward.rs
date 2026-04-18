use std::slice;
use wasm_bindgen::prelude::*;

/// Relu backward: passes gradient where the forward input was positive, zeros elsewhere.
/// Takes contiguous inputs — the autograd engine always allocates fresh tensors,
/// so we use a simple length-based signature (no meta buffer).
///
/// inputs: grad, a (original forward input)
#[wasm_bindgen]
pub fn relu_backward_raw(grad_ptr: *const f32, a_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    unsafe {
        let grad = slice::from_raw_parts(grad_ptr, len);
        let a = slice::from_raw_parts(a_ptr, len);
        let out = slice::from_raw_parts_mut(out_ptr, len);
        for i in 0..len {
            out[i] = if a[i] > 0.0 { grad[i] } else { 0.0 };
        }
    }
}
