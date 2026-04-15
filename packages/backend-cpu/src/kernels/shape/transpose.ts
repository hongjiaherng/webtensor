import { CPUKernel } from '../utils';

export function executeTranspose(
  a: Float32Array, out: Float32Array,
  m: number, n: number,
): void {
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      out[col * m + row] = a[row * n + col];
    }
  }
}

export const transposeKernel: CPUKernel = (_node, inputs, outputs) => {
  const shape = inputs[0].shape as number[];
  const m = shape[shape.length - 2] || 1;
  const n = shape[shape.length - 1];
  executeTranspose(
    inputs[0].storage.buffer as Float32Array,
    outputs[0].storage.buffer as Float32Array,
    m, n,
  );
};
