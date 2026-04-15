// Generic strided MatMul kernel (2-D only: A[M,K] × B[K,N] = Out[M,N]).
//
// Inputs A and B may have arbitrary strides and offsets (e.g. a transposed
// view).  The output is always written contiguously row-major.
//
// TensorMeta uniform — same 80-byte layout as all other kernels:
//   rank, offset, padding×2, shape[0..7] (2×vec4), strides[0..7] (2×vec4)
//
// For 2-D tensors: shape[0]=rows, shape[1]=cols
//                  strides[0]=row_stride, strides[1]=col_stride

struct TensorMeta {
  rank:    u32,
  offset:  u32,
  _p0:     u32,
  _p1:     u32,
  shape:   array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
};

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read>       b:      array<f32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;
@group(0) @binding(3) var<uniform>             meta_a: TensorMeta;
@group(0) @binding(4) var<uniform>             meta_b: TensorMeta;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;

  let M = meta_a.shape[0u][0u];   // A rows
  let K = meta_a.shape[0u][1u];   // A cols = B rows
  let N = meta_b.shape[0u][1u];   // B cols

  if (row >= M || col >= N) { return; }

  let a_row_stride = meta_a.strides[0u][0u];
  let a_col_stride = meta_a.strides[0u][1u];
  let b_row_stride = meta_b.strides[0u][0u];
  let b_col_stride = meta_b.strides[0u][1u];
  let a_off        = meta_a.offset;
  let b_off        = meta_b.offset;

  var sum = 0.0f;
  for (var k = 0u; k < K; k++) {
    let a_idx = a_off + row * a_row_stride + k * a_col_stride;
    let b_idx = b_off + k   * b_row_stride + col * b_col_stride;
    sum += a[a_idx] * b[b_idx];
  }

  out[row * N + col] = sum;
}
