@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let index = global_id.x;
  
  let num_elements_out = arrayLength(&Out);
  if (index >= num_elements_out) {
    return;
  }

  let lenA = arrayLength(&A);
  let lenB = arrayLength(&B);
  
  // Pseudo-broadcasting: if length is 1, treat it as a broadcast scalar
  let indexA = select(index, 0u, lenA == 1u);
  let indexB = select(index, 0u, lenB == 1u);

  Out[index] = A[indexA] + B[indexB];
}
