// Strided ReduceMean. Same as ReduceSum but divides by reduce_total.

__TENSOR_META__

struct ReduceMeta {
  kept_rank:    u32,
  reduce_rank:  u32,
  kept_total:   u32,
  reduce_total: u32,
  kept_axes:    array<u32, 64>,
  reduce_axes:  array<u32, 64>,
};

@group(0) @binding(0) var<storage, read>       a:        array<f32>;
@group(0) @binding(1) var<storage, read_write> out:      array<f32>;
@group(0) @binding(2) var<uniform>             u_meta_a: TensorMeta;
@group(0) @binding(3) var<uniform>             u_reduce: ReduceMeta;

fn shape_of(ax: u32) -> u32 { return u_meta_a.shape[ax]; }
fn stride_of(ax: u32) -> u32 { return u_meta_a.strides[ax]; }
fn kept_axis(i: u32) -> u32 { return u_reduce.kept_axes[i]; }
fn reduce_axis(i: u32) -> u32 { return u_reduce.reduce_axes[i]; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>) {
  let out_idx = gid.y * ng.x * 64u + gid.x;
  if (out_idx >= u_reduce.kept_total) { return; }

  var coord: array<u32, 64>;
  for (var i = 0u; i < 64u; i++) { coord[i] = 0u; }

  var rem = out_idx;
  for (var d = u_reduce.kept_rank; d > 0u; d--) {
    let i = d - 1u;
    let ax = kept_axis(i);
    let dim = shape_of(ax);
    coord[ax] = rem % dim;
    rem = rem / dim;
  }

  var acc = 0.0f;
  for (var r = 0u; r < u_reduce.reduce_total; r++) {
    var r_rem = r;
    for (var d = u_reduce.reduce_rank; d > 0u; d--) {
      let i = d - 1u;
      let ax = reduce_axis(i);
      let dim = shape_of(ax);
      coord[ax] = r_rem % dim;
      r_rem = r_rem / dim;
    }
    var off = u_meta_a.offset;
    for (var d = 0u; d < u_meta_a.rank; d++) {
      off += coord[d] * stride_of(d);
    }
    acc += a[off];
  }

  let n = f32(u_reduce.reduce_total);
  out[out_idx] = select(0.0f, acc / n, n > 0.0f);
}
