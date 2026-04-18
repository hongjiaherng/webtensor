use std::slice;
use wasm_bindgen::prelude::*;

/// Batched strided matmul for float32. Broadcasts over leading batch dims.
///
/// meta layout (32 × u32):
///   [0]       batch_rank     (0..=6)
///   [1]       M
///   [2]       K
///   [3]       N
///   [4]       a_row_stride
///   [5]       a_col_stride
///   [6]       b_row_stride
///   [7]       b_col_stride
///   [8]       a_offset
///   [9]       b_offset
///   [10..16]  batch_out_shape[0..6]
///   [16..22]  a_bcast_batch_strides[0..6]   (broadcast-aligned to batch_out_shape)
///   [22..28]  b_bcast_batch_strides[0..6]
///   [28..32]  padding
#[wasm_bindgen]
pub fn matmul_strided(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    meta_ptr: *const u32,
) {
    unsafe {
        let meta = slice::from_raw_parts(meta_ptr, 32);
        let batch_rank = meta[0] as usize;
        let m = meta[1] as usize;
        let k = meta[2] as usize;
        let n = meta[3] as usize;
        let a_row_stride = meta[4] as usize;
        let a_col_stride = meta[5] as usize;
        let b_row_stride = meta[6] as usize;
        let b_col_stride = meta[7] as usize;
        let a_off = meta[8] as usize;
        let b_off = meta[9] as usize;

        let batch_out_shape = &meta[10..10 + batch_rank];
        let a_bcast = &meta[16..16 + batch_rank];
        let b_bcast = &meta[22..22 + batch_rank];

        let mut batch_total: usize = 1;
        for i in 0..batch_rank {
            batch_total *= batch_out_shape[i] as usize;
        }

        let out_mat_stride = m * n;
        let out = slice::from_raw_parts_mut(out_ptr, batch_total * out_mat_stride);

        let mut coord = [0u32; 6];
        for b_idx in 0..batch_total {
            let mut rem = b_idx;
            for i in (0..batch_rank).rev() {
                let d = batch_out_shape[i] as usize;
                coord[i] = (rem % d) as u32;
                rem /= d;
            }

            let mut a_batch_off: usize = 0;
            let mut b_batch_off: usize = 0;
            for i in 0..batch_rank {
                a_batch_off += (coord[i] * a_bcast[i]) as usize;
                b_batch_off += (coord[i] * b_bcast[i]) as usize;
            }

            let a_base = a_off + a_batch_off;
            let b_base = b_off + b_batch_off;
            let out_base = b_idx * out_mat_stride;

            for row in 0..m {
                for col in 0..n {
                    let mut sum: f32 = 0.0;
                    for ki in 0..k {
                        let ai = a_base + row * a_row_stride + ki * a_col_stride;
                        let bi = b_base + ki * b_row_stride + col * b_col_stride;
                        sum += *a_ptr.add(ai) * *b_ptr.add(bi);
                    }
                    out[out_base + row * n + col] = sum;
                }
            }
        }
    }
}
