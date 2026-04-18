__TENSOR_META__

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read_write> out:    array<f32>;
@group(0) @binding(2) var<uniform>             u_meta: TensorMeta;
@group(0) @binding(3) var<uniform>             u_exp:  f32;

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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  let base = a[strided_idx(i)];
  // WGSL `pow(x, y)` is undefined for negative x. Use |x|^y, then flip the
  // sign when the base is negative and the exponent is an odd integer
  // (so e.g. `pow(-2, 2) = 4` and `pow(-2, 3) = -8`).
  let abs_base = abs(base);
  let result = pow(abs_base, u_exp);
  let n = round(u_exp);
  let is_int = abs(u_exp - n) < 1e-6;
  let is_odd = (i32(n) & 1) == 1;
  let flip = base < 0.0 && is_int && is_odd;
  out[i] = select(result, -result, flip);
}
