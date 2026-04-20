// Concat — scatter one input into the output along `u_concat.axis`, starting
// at `u_concat.axis_start`. One thread per input element: unravel the input
// flat index against the input shape, shift the axis coord by `axis_start`,
// then re-ravel against the output (contiguous) strides.
//
// The backend dispatches this shader once per input (see concat.ts). SCALAR is
// substituted per output dtype. u32 covers bool on device.

__TENSOR_META__

struct ConcatMeta {
  axis:       u32,
  axis_start: u32,
  _p0:        u32,
  _p1:        u32,
};

@group(0) @binding(0) var<storage, read>       a:          array<SCALAR>;
@group(0) @binding(1) var<storage, read_write> out:        array<SCALAR>;
@group(0) @binding(2) var<uniform>             u_meta_a:   TensorMeta;
@group(0) @binding(3) var<uniform>             u_meta_out: TensorMeta;
@group(0) @binding(4) var<uniform>             u_concat:   ConcatMeta;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 64u + gid.x;

  // Input element count = product of input shape.
  let rank = u_meta_a.rank;
  var in_total: u32 = 1u;
  for (var d: u32 = 0u; d < rank; d = d + 1u) {
    in_total = in_total * u_meta_a.shape[d];
  }
  if (i >= in_total) { return; }

  // Unravel i against input shape; re-ravel against output contiguous strides,
  // shifting the axis coord by axis_start.
  var rem = i;
  var out_flat: u32 = 0u;
  for (var d: u32 = rank; d > 0u; d = d - 1u) {
    let ax = d - 1u;
    let in_dim = u_meta_a.shape[ax];
    var coord = rem % in_dim;
    rem = rem / in_dim;
    if (ax == u_concat.axis) {
      coord = coord + u_concat.axis_start;
    }
    out_flat = out_flat + coord * u_meta_out.strides[ax];
  }

  out[out_flat] = a[strided_idx_a(i)];
}
