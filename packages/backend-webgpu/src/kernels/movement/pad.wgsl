// Constant-value Pad. Gather-style: one thread per output element. Unravel
// the output flat index against the output shape; for each axis, subtract
// `pads_before[ax]` to recover the input coord. If any adjusted coord is
// outside `[0, input_shape[ax])`, write the fill value (bit-pattern encoded so
// the same shader works for f32 / i32 / u32). Otherwise, ravel the adjusted
// coord against the input strides and copy.

__TENSOR_META__

struct PadMeta {
  pads_before: array<u32, 64>,
  value_bits:  u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read>       a:          array<SCALAR>;
@group(0) @binding(1) var<storage, read_write> out:        array<SCALAR>;
@group(0) @binding(2) var<uniform>             u_meta_a:   TensorMeta;
@group(0) @binding(3) var<uniform>             u_meta_out: TensorMeta;
@group(0) @binding(4) var<uniform>             u_pad:      PadMeta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
  let i = gid.y * ng.x * 64u + gid.x;
  if (i >= arrayLength(&out)) { return; }

  let rank = u_meta_out.rank;
  var rem = i;
  var in_flat: u32 = u_meta_a.offset;
  var in_bounds: bool = true;
  for (var d: u32 = rank; d > 0u; d = d - 1u) {
    let ax = d - 1u;
    let out_dim = u_meta_out.shape[ax];
    let out_coord = rem % out_dim;
    rem = rem / out_dim;

    let pad_before = u_pad.pads_before[ax];
    if (out_coord < pad_before) {
      in_bounds = false;
    } else {
      let in_coord = out_coord - pad_before;
      let in_dim = u_meta_a.shape[ax];
      if (in_coord >= in_dim) {
        in_bounds = false;
      } else {
        in_flat = in_flat + in_coord * u_meta_a.strides[ax];
      }
    }
  }

  if (in_bounds) {
    out[i] = a[in_flat];
  } else {
    out[i] = bitcast<SCALAR>(u_pad.value_bits);
  }
}
