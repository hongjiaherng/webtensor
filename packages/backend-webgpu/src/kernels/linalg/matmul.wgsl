// Batched strided MatMul kernel.
// A shape: [...batchOut, M, K]   (packed with broadcast-aligned batch strides)
// B shape: [...batchOut, K, N]   (packed with broadcast-aligned batch strides)
// Out: contiguous [...batchOut, M, N]

__TENSOR_META__

struct BatchMeta {
  batch_rank: u32,
  M:          u32,
  K:          u32,
  N:          u32,
  batch_out_shape: array<u32, __MAX_RANK__>,
};

@group(0) @binding(0) var<storage, read>       a:        array<f32>;
@group(0) @binding(1) var<storage, read>       b:        array<f32>;
@group(0) @binding(2) var<storage, read_write> out:      array<f32>;
@group(0) @binding(3) var<uniform>             u_meta_a: TensorMeta;
@group(0) @binding(4) var<uniform>             u_meta_b: TensorMeta;
@group(0) @binding(5) var<uniform>             u_batch:  BatchMeta;

fn a_shape(ax: u32) -> u32 { return u_meta_a.shape[ax]; }
fn a_stride(ax: u32) -> u32 { return u_meta_a.strides[ax]; }
fn b_stride(ax: u32) -> u32 { return u_meta_b.strides[ax]; }
fn batch_shape(ax: u32) -> u32 { return u_batch.batch_out_shape[ax]; }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  let b_idx = gid.z;
  let M = u_batch.M;
  let K = u_batch.K;
  let N = u_batch.N;
  let br = u_batch.batch_rank;

  if (row >= M || col >= N) { return; }

  var batch_total: u32 = 1u;
  for (var i = 0u; i < br; i++) { batch_total *= batch_shape(i); }
  if (b_idx >= max(batch_total, 1u)) { return; }

  // Decompose b_idx over batch_out_shape (innermost last)
  var rem = b_idx;
  var a_base: u32 = u_meta_a.offset;
  var b_base: u32 = u_meta_b.offset;
  for (var d = br; d > 0u; d--) {
    let ax = d - 1u;
    let dim = batch_shape(ax);
    let coord = rem % dim;
    rem = rem / dim;
    a_base += coord * a_stride(ax);
    b_base += coord * b_stride(ax);
  }

  let a_row_s = a_stride(br);
  let a_col_s = a_stride(br + 1u);
  let b_row_s = b_stride(br);
  let b_col_s = b_stride(br + 1u);

  var sum = 0.0f;
  for (var k = 0u; k < K; k++) {
    let ai = a_base + row * a_row_s + k * a_col_s;
    let bi = b_base + k * b_row_s + col * b_col_s;
    sum += a[ai] * b[bi];
  }

  let out_mat = M * N;
  out[b_idx * out_mat + row * N + col] = sum;
}
