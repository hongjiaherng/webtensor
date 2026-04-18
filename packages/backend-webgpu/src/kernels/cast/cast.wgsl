// Strided dtype cast. IN_SCALAR and OUT_SCALAR are substituted at pipeline-build
// time (f32 / i32). Bool is not supported on WebGPU: bool tensors are stored as
// 1 B/elem on the host but WGSL has no 1-byte scalar — the host/device layouts
// don't match. Cast bool tensors on CPU or WASM first.
//
// Numeric conversion is `OUT_SCALAR(v)` — WGSL's explicit type conversion,
// which truncates f32 → i32 toward zero (matches NumPy / PyTorch).

struct TensorMeta {
  rank:    u32,
  offset:  u32,
  _p0:     u32,
  _p1:     u32,
  shape:   array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
};

@group(0) @binding(0) var<storage, read>       a:      array<IN_SCALAR>;
@group(0) @binding(1) var<storage, read_write> out:    array<OUT_SCALAR>;
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
  out[i] = OUT_SCALAR(a[strided_idx(i)]);
}
