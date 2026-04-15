// Generic strided Relu kernel.
// Input may have arbitrary strides and offset; output is always contiguous.

struct TensorMeta {
  rank:    u32,
  offset:  u32,
  _p0:     u32,
  _p1:     u32,
  shape:   array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
};

@group(0) @binding(0) var<storage, read>       a:    array<f32>;
@group(0) @binding(1) var<storage, read_write> out:  array<f32>;
@group(0) @binding(2) var<uniform>             u_meta: TensorMeta;

fn strided_idx(flat: u32) -> u32 {
  let rank = u_meta.rank;
  var rem = flat;
  var idx = u_meta.offset;
  for (var d = rank; d > 0u; d--) {
    let ax  = d - 1u;
    let dim = u_meta.shape[ax / 4u][ax % 4u];
    let s   = u_meta.strides[ax / 4u][ax % 4u];
    idx += (rem % dim) * s;
    rem  /= dim;
  }
  return idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  out[i] = max(0.0, a[strided_idx(i)]);
}
