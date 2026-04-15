import { CPUKernel } from '../utils';

export function executeSub(a: Float32Array, b: Float32Array, out: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = a[i % a.length] - b[i % b.length];
  }
}

export const subKernel: CPUKernel = (_node, inputs, outputs) => {
  executeSub(inputs[0].buffer as Float32Array, inputs[1].buffer as Float32Array, outputs[0].buffer as Float32Array);
};
