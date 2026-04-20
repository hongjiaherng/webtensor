// Element-wise `|a - b| <= atol + rtol * |b|`. Float32-only. Tolerances and
// equal_nan arrive in a small `ToleranceMeta` uniform at binding 5. NaN/NaN is
// equal only when `equal_nan == 1`. ±inf compares equal only to itself
// (without this guard, `|inf - (-inf)| <= atol + rtol*inf` evaluates to true).

__TENSOR_META__

struct ToleranceMeta {
  rtol: f32,
  atol: f32,
  equal_nan: u32,
  _p0: u32,
};

@group(0) @binding(0) var<storage, read>       a:        array<f32>;
@group(0) @binding(1) var<storage, read>       b:        array<f32>;
@group(0) @binding(2) var<storage, read_write> out:      array<u32>;
@group(0) @binding(3) var<uniform>             u_meta_a: TensorMeta;
@group(0) @binding(4) var<uniform>             u_meta_b: TensorMeta;
@group(0) @binding(5) var<uniform>             u_tol:    ToleranceMeta;

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

fn is_nan_f32(v: f32) -> bool { return v != v; }

fn is_inf_f32(v: f32) -> bool {
  let bits = bitcast<u32>(v);
  return (bits & 0x7fffffffu) == 0x7f800000u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 64u + gid.x;
  if (i >= arrayLength(&out)) { return; }
  let av = a[strided_idx_a(i)];
  let bv = b[strided_idx_b(i)];

  var close: bool;
  let a_nan = is_nan_f32(av);
  let b_nan = is_nan_f32(bv);
  if (a_nan || b_nan) {
    close = (u_tol.equal_nan == 1u) && a_nan && b_nan;
  } else if (is_inf_f32(av) || is_inf_f32(bv)) {
    close = av == bv;
  } else {
    close = abs(av - bv) <= u_tol.atol + u_tol.rtol * abs(bv);
  }
  out[i] = select(0u, 1u, close);
}
