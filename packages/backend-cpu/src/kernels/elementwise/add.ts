import { CPUKernel } from '../utils';

export function executeAdd(a: Float32Array, b: Float32Array, out: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = a[i % a.length] + b[i % b.length];
  }
}

export const addKernel: CPUKernel = (_node, inputs, outputs) => {
  executeAdd(inputs[0].storage.buffer as Float32Array, inputs[1].storage.buffer as Float32Array, outputs[0].storage.buffer as Float32Array);
};
