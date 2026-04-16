// Generic strided Add kernel.
//
// Inputs A and B may have arbitrary strides and offsets (including broadcast
// dimensions where stride == 0).  The output is always written contiguously.
//
// TensorMeta uniform layout (80 bytes):
//   [u32  0] rank
//   [u32  1] offset (element offset into the buffer)
//   [u32  2] padding
//   [u32  3] padding
//   [u32  4..11] shape[0..7]   — packed as 2 × vec4<u32>
//   [u32 12..19] strides[0..7] — packed as 2 × vec4<u32>
//
// For binary ops the shape stored in each meta is the *output* shape.
// Broadcast dimensions carry stride == 0 so the same element is reused.

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
@group(0) @binding(3) var<uniform>             u_meta_a: TensorMeta;
@group(0) @binding(4) var<uniform>             u_meta_b: TensorMeta;

fn strided_idx_a(flat: u32) -> u32 {
  let rank = u_meta_a.rank;
  var rem = flat;
  var idx = u_meta_a.offset;
  for (var d = rank; d > 0u; d--) {
    let ax  = d - 1u;
    let dim = u_meta_a.shape[ax / 4u][ax % 4u];
    let s   = u_meta_a.strides[ax / 4u][ax % 4u];
    idx += (rem % dim) * s;
    rem  /= dim;
  }
  return idx;
}

fn strided_idx_b(flat: u32) -> u32 {
  let rank = u_meta_b.rank;
  var rem = flat;
  var idx = u_meta_b.offset;
  for (var d = rank; d > 0u; d--) {
    let ax  = d - 1u;
    let dim = u_meta_b.shape[ax / 4u][ax % 4u];
    let s   = u_meta_b.strides[ax / 4u][ax % 4u];
    idx += (rem % dim) * s;
    rem  /= dim;
  }
  return idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  out[i] = a[strided_idx_a(i)] + b[strided_idx_b(i)];
}
