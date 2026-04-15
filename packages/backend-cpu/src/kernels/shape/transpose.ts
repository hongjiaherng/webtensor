export function executeTranspose(
  a: Float32Array, out: Float32Array,
  m: number, n: number
): void {
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      out[col * m + row] = a[row * n + col];
    }
  }
}
