use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn alloc_f32(len: usize) -> *mut f32 {
    let mut buffer = Vec::<f32>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

#[wasm_bindgen]
pub unsafe fn free_f32(ptr: *mut f32, len: usize) {
    if ptr.is_null() {
        return;
    }
    drop(Vec::from_raw_parts(ptr, 0, len));
}
