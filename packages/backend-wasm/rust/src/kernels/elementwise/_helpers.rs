//! Shared macro for element-wise comparison kernels (eq / ne / lt / le / gt / ge).
//! Inputs share one arithmetic dtype (f32 or i32); output is u8 (bool 0/1).
//! Meta layout: see `kernels::BINARY_META_WORDS`.

/// Generate a strided broadcast comparison: apply `$pred(a, b) -> bool`, write
/// the result to a u8 output as 0 / 1.
#[macro_export]
macro_rules! compare_kernel {
    ($name:ident, $scalar:ty, |$a:ident, $b:ident| $pred:expr) => {
        #[wasm_bindgen::prelude::wasm_bindgen]
        pub fn $name(
            a_ptr: *const $scalar,
            b_ptr: *const $scalar,
            out_ptr: *mut u8,
            meta_ptr: *const u32,
        ) {
            unsafe {
                let meta = std::slice::from_raw_parts(
                    meta_ptr,
                    $crate::kernels::BINARY_META_WORDS,
                );
                let total = meta[0] as usize;
                let rank = meta[1] as usize;
                let shape = &meta[$crate::kernels::BINARY_SHAPE_OFF
                    ..$crate::kernels::BINARY_SHAPE_OFF + rank];
                let a_bcast = &meta[$crate::kernels::BINARY_A_STRIDES_OFF
                    ..$crate::kernels::BINARY_A_STRIDES_OFF + rank];
                let a_off = meta[$crate::kernels::BINARY_A_OFFSET_OFF];
                let b_bcast = &meta[$crate::kernels::BINARY_B_STRIDES_OFF
                    ..$crate::kernels::BINARY_B_STRIDES_OFF + rank];
                let b_off = meta[$crate::kernels::BINARY_B_OFFSET_OFF];
                let out = std::slice::from_raw_parts_mut(out_ptr, total);
                for i in 0..total {
                    let ai = $crate::utils::strided_idx(shape, a_bcast, a_off, i as u32);
                    let bi = $crate::utils::strided_idx(shape, b_bcast, b_off, i as u32);
                    let $a: $scalar = *a_ptr.add(ai);
                    let $b: $scalar = *b_ptr.add(bi);
                    out[i] = if $pred { 1 } else { 0 };
                }
            }
        }
    };
}
