// Shared template for element-wise broadcast comparisons. SCALAR is the input
// dtype (f32 / i32); the output is always u32 (0/1) for bool. PRED is the
// per-element predicate expression over `av` and `bv`, substituted at
// pipeline-build time (e.g. `av == bv`, `av < bv`).

__TENSOR_META__

@group(0) @binding(0) var<storage, read>       a:        array<SCALAR>;
@group(0) @binding(1) var<storage, read>       b:        array<SCALAR>;
@group(0) @binding(2) var<storage, read_write> out:      array<u32>;
@group(0) @binding(3) var<uniform>             u_meta_a: TensorMeta;
@group(0) @binding(4) var<uniform>             u_meta_b: TensorMeta;

fn strided_idx_a(flat: u32) -> u32 {
  let rank = u_meta_a.rank;
  var rem = flat;
  var idx = u_meta_a.offset;
  for (var d = rank; d > 0u; d--) {
    let ax  = d - 1u;
    let dim = u_meta_a.shape[ax];
    let s   = u_meta_a.strides[ax];
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
    let dim = u_meta_b.shape[ax];
    let s   = u_meta_b.strides[ax];
    idx += (rem % dim) * s;
    rem  /= dim;
  }
  return idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  let av = a[strided_idx_a(i)];
  let bv = b[strided_idx_b(i)];
  out[i] = select(0u, 1u, PRED);
}
