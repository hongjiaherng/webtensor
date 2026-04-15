struct Uniforms {
  M: u32,
  K: u32,
  N: u32,
  padding: u32,
};

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // global_id.x maps to row, global_id.y maps to col 
  // (Assuming X acts on M axis and Y acts on N axis)
  let row = global_id.x;
  let col = global_id.y;

  if (row >= uniforms.M || col >= uniforms.N) {
    return;
  }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < uniforms.K; k = k + 1u) {
    let a_idx = row * uniforms.K + k;      // A is [M, K]
    let b_idx = k * uniforms.N + col;      // B is [K, N]
    sum = sum + A[a_idx] * B[b_idx];
  }

  let out_idx = row * uniforms.N + col;    // Out is [M, N]
  Out[out_idx] = sum;
}
