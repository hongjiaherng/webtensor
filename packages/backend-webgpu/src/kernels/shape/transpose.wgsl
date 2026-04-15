@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

// Utilize the existing 16 byte unified meta buffer
struct Uniforms {
  M: u32,
  N: u32,
  PADDING_1: u32,
  PADDING_2: u32,
};
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let id = global_id.x;
    let total = uniforms.M * uniforms.N;
    if (id >= total) {
        return;
    }
    
    // id = flat index in the input.
    // In a 2D matrix (M x N) mapped linearly:
    let m = id / uniforms.N;
    let n = id % uniforms.N;
    
    // Transposed out index maps to (n, m) inside an N x M matrix
    let out_idx = n * uniforms.M + m;
    output[out_idx] = input[id];
}
