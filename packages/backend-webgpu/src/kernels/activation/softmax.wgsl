// Strided softmax. One invocation per "slice" (non-axis index).
// Output is contiguous with same shape as input.

__TENSOR_META__

struct SoftmaxMeta {
  axis:        u32,
  slice_count: u32,
  axis_len:    u32,
  _pad:        u32,
  out_strides: array<u32, 64>,
};

@group(0) @binding(0) var<storage, read>       a:        array<f32>;
@group(0) @binding(1) var<storage, read_write> out:      array<f32>;
@group(0) @binding(2) var<uniform>             u_meta_a: TensorMeta;
@group(0) @binding(3) var<uniform>             u_sm:     SoftmaxMeta;

fn shape_of(ax: u32) -> u32 { return u_meta_a.shape[ax]; }
fn stride_of(ax: u32) -> u32 { return u_meta_a.strides[ax]; }
fn out_stride_of(ax: u32) -> u32 { return u_sm.out_strides[ax]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= u_sm.slice_count) { return; }

  var coord: array<u32, 64>;
  for (var i = 0u; i < 64u; i++) { coord[i] = 0u; }

  var rem = s;
  for (var d = u_meta_a.rank; d > 0u; d--) {
    let ax = d - 1u;
    if (ax == u_sm.axis) { continue; }
    let dim = shape_of(ax);
    coord[ax] = rem % dim;
    rem = rem / dim;
  }

  var in_base = u_meta_a.offset;
  var out_base: u32 = 0u;
  for (var d = 0u; d < u_meta_a.rank; d++) {
    in_base += coord[d] * stride_of(d);
    out_base += coord[d] * out_stride_of(d);
  }
  let in_axis_stride = stride_of(u_sm.axis);
  let out_axis_stride = out_stride_of(u_sm.axis);

  // Pass 1: max
  var max_v = a[in_base];
  for (var k = 1u; k < u_sm.axis_len; k++) {
    let v = a[in_base + k * in_axis_stride];
    if (v > max_v) { max_v = v; }
  }

  // Pass 2: exp + sum
  var sum = 0.0f;
  for (var k = 0u; k < u_sm.axis_len; k++) {
    let e = exp(a[in_base + k * in_axis_stride] - max_v);
    out[out_base + k * out_axis_stride] = e;
    sum += e;
  }

  // Pass 3: divide
  let inv_sum = select(0.0f, 1.0f / sum, sum > 0.0f);
  for (var k = 0u; k < u_sm.axis_len; k++) {
    out[out_base + k * out_axis_stride] = out[out_base + k * out_axis_stride] * inv_sum;
  }
}
