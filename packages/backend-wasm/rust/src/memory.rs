use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn alloc_f32(len: usize) -> *mut f32 {
    let mut buffer = Vec::<f32>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

#[wasm_bindgen]
pub fn free_f32(ptr: *mut f32, len: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: ptr was allocated by alloc_f32 with the same capacity.
    // Reconstructing the Vec transfers ownership back to Rust so it is freed on drop.
    unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// Allocate a block of `len` u32 values for passing shape/stride meta buffers
/// from JavaScript into strided kernels.
#[wasm_bindgen]
pub fn alloc_u32(len: usize) -> *mut u32 {
    let mut buffer = Vec::<u32>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

#[wasm_bindgen]
pub fn free_u32(ptr: *mut u32, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}
