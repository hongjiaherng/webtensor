// contiguous.wgsl — copies a contiguous (post-gather) input to the output.
// The gather pre-pass in WebGPUBackend.execute() has already materialized any
// non-contiguous strides, so this kernel is a plain element-wise copy.
// arrayLength() avoids a meta/uniform buffer entirely.

@group(0) @binding(0) var<storage, read>       inp : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  out[i] = inp[i];
}
