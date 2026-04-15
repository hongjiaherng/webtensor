use std::slice;
use wasm_bindgen::prelude::*;

/// Strided 2-D matrix multiply.
///
/// meta layout (9 × u32):
///   [0]  M
///   [1]  K
///   [2]  N
///   [3]  a_row_stride   (A.strides[rank-2])
///   [4]  a_col_stride   (A.strides[rank-1])
///   [5]  b_row_stride
///   [6]  b_col_stride
///   [7]  a_offset
///   [8]  b_offset
#[wasm_bindgen]
pub fn matmul_strided(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    meta_ptr: *const u32,
) {
    unsafe {
        let meta = slice::from_raw_parts(meta_ptr, 9);
        let m = meta[0] as usize;
        let k = meta[1] as usize;
        let n = meta[2] as usize;
        let a_row_stride = meta[3] as usize;
        let a_col_stride = meta[4] as usize;
        let b_row_stride = meta[5] as usize;
        let b_col_stride = meta[6] as usize;
        let a_off = meta[7] as usize;
        let b_off = meta[8] as usize;
        let out = slice::from_raw_parts_mut(out_ptr, m * n);

        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0_f32;
                for ki in 0..k {
                    let ai = a_off + row * a_row_stride + ki * a_col_stride;
                    let bi = b_off + ki * b_row_stride + col * b_col_stride;
                    sum += *a_ptr.add(ai) * *b_ptr.add(bi);
                }
                out[row * n + col] = sum;
            }
        }
    }
}
