export function executeRelu(a: Float32Array, out: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = a[i] > 0 ? a[i] : 0;
  }
}

// Backward: passes gradient where input was positive, zeros it elsewhere.
// grad * (a > 0 ? 1 : 0)
export function executeReluGrad(grad: Float32Array, a: Float32Array, out: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = a[i] > 0 ? grad[i] : 0;
  }
}
