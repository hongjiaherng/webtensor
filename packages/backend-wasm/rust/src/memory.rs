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
    unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

#[wasm_bindgen]
pub fn alloc_i32(len: usize) -> *mut i32 {
    let mut buffer = Vec::<i32>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

#[wasm_bindgen]
pub fn free_i32(ptr: *mut i32, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

#[wasm_bindgen]
pub fn alloc_u8(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

#[wasm_bindgen]
pub fn free_u8(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// u32 allocation used for meta buffers (shape/stride pointers).
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
