import { CPUKernel } from '../utils';

export function executeMul(a: Float32Array, b: Float32Array, out: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = a[i % a.length] * b[i % b.length];
  }
}

export const mulKernel: CPUKernel = (_node, inputs, outputs) => {
  executeMul(inputs[0].buffer as Float32Array, inputs[1].buffer as Float32Array, outputs[0].buffer as Float32Array);
};
