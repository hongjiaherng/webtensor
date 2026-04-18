// Generic strided Contiguous kernel.
// Reads from a potentially non-contiguous input (any strides/offset) and
// writes to a packed contiguous output buffer.

__TENSOR_META__

@group(0) @binding(0) var<storage, read>       inp:  array<f32>;
@group(0) @binding(1) var<storage, read_write> out:  array<f32>;
@group(0) @binding(2) var<uniform>             u_meta: TensorMeta;

fn strided_idx(flat: u32) -> u32 {
  let rank = u_meta.rank;
  var rem = flat;
  var idx = u_meta.offset;
  for (var d = rank; d > 0u; d--) {
    let ax  = min(d - 1u, 63u);
    let dim = u_meta.shape[ax];
    let s   = u_meta.strides[ax];
    idx += (rem % dim) * s;
    rem  /= dim;
  }
  return idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  out[i] = inp[strided_idx(i)];
}
