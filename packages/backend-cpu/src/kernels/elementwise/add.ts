export function executeAdd(a: Float32Array, b: Float32Array, out: Float32Array): void {
  const aScalar = a.length === 1;
  const bScalar = b.length === 1;
  for (let i = 0; i < out.length; i++) {
    const valA = aScalar ? a[0] : a[i];
    const valB = bScalar ? b[0] : b[i];
    out[i] = valA + valB;
  }
}
