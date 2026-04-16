// ReluGrad: out[i] = a[i] > 0 ? grad[i] : 0
// inputs[0] = grad (upstream gradient), inputs[1] = a (original relu input)
// Both inputs may have arbitrary strides and offset; output is always contiguous.

struct TensorMeta {
  rank:    u32,
  offset:  u32,
  _p0:     u32,
  _p1:     u32,
  shape:   array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
};

@group(0) @binding(0) var<storage, read>       grad_in: array<f32>;
@group(0) @binding(1) var<storage, read>       a_in:    array<f32>;
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;
@group(0) @binding(3) var<uniform>             u_meta_grad: TensorMeta;
@group(0) @binding(4) var<uniform>             u_meta_a:    TensorMeta;

fn strided_idx(m: TensorMeta, flat: u32) -> u32 {
  let rank = m.rank;
  var rem = flat;
  var idx = m.offset;
  for (var d = rank; d > 0u; d--) {
    let ax  = d - 1u;
    let dim = m.shape[ax / 4u][ax % 4u];
    let s   = m.strides[ax / 4u][ax % 4u];
    idx += (rem % dim) * s;
    rem  /= dim;
  }
  return idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  let g = grad_in[strided_idx(u_meta_grad, i)];
  let a = a_in[strided_idx(u_meta_a, i)];
  out[i] = select(0.0, g, a > 0.0);
}
