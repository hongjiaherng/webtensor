export function executeSub(a: Float32Array, b: Float32Array, out: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = a[i % a.length] - b[i % b.length];
  }
}
