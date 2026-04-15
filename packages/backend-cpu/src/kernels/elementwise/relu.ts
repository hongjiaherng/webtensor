import { CPUKernel } from '../utils';

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

export const reluKernel: CPUKernel = (_node, inputs, outputs) => {
  executeRelu(inputs[0].storage.buffer as Float32Array, outputs[0].storage.buffer as Float32Array);
};

export const reluGradKernel: CPUKernel = (_node, inputs, outputs) => {
  executeReluGrad(
    inputs[0].storage.buffer as Float32Array,  // grad
    inputs[1].storage.buffer as Float32Array,  // original input a
    outputs[0].storage.buffer as Float32Array,
  );
};
