export function executeMatMul(
  a: Float32Array, b: Float32Array, out: Float32Array,
  m: number, k: number, n: number
): void {
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0;
      for (let i = 0; i < k; i++) {
        sum += a[row * k + i] * b[i * n + col];
      }
      out[row * n + col] = sum;
    }
  }
}
