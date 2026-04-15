import { CPUKernel } from '../utils';

export function executeMatMul(
  a: Float32Array, b: Float32Array, out: Float32Array,
  m: number, k: number, n: number,
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

export const matmulKernel: CPUKernel = (_node, inputs, outputs) => {
  const shapeA = inputs[0].shape as number[];
  const shapeB = inputs[1].shape as number[];
  const m = shapeA[shapeA.length - 2] || 1;
  const k = shapeA[shapeA.length - 1];
  const n = shapeB[shapeB.length - 1];
  executeMatMul(
    inputs[0].buffer as Float32Array,
    inputs[1].buffer as Float32Array,
    outputs[0].buffer as Float32Array,
    m, k, n,
  );
};
