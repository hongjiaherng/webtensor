// Strided dtype cast. IN_SCALAR / OUT_SCALAR / CAST_EXPR are substituted at
// pipeline-build time. IN_SCALAR and OUT_SCALAR are one of f32 / i32 / u32
// (bool is stored as u32 on device, see backend.ts). CAST_EXPR is the per-
// element expression:
//   - bool out  : select(0u, 1u, v != IN_SCALAR(0))   (truthiness)
//   - else      : OUT_SCALAR(v)                        (native WGSL coercion —
//                                                       f32 → i32 truncates
//                                                       toward zero, matching
//                                                       NumPy / PyTorch)

__TENSOR_META__

@group(0) @binding(0) var<storage, read>       a:      array<IN_SCALAR>;
@group(0) @binding(1) var<storage, read_write> out:    array<OUT_SCALAR>;
@group(0) @binding(2) var<uniform>             u_meta: TensorMeta;

fn strided_idx(flat: u32) -> u32 {
  let rank = u_meta.rank;
  var rem = flat;
  var idx = u_meta.offset;
  for (var d = rank; d > 0u; d--) {
    let ax  = d - 1u;
    let dim = u_meta.shape[ax];
    let s   = u_meta.strides[ax];
    idx += (rem % dim) * s;
    rem  /= dim;
  }
  return idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 64u + gid.x;
  if (i >= arrayLength(&out)) { return; }
  let v = a[strided_idx(i)];
  out[i] = CAST_EXPR;
}
