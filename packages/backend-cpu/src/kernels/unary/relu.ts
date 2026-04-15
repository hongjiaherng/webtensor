import { CPUKernel, stridedIdx } from '../utils';

export const reluKernel: CPUKernel = (_node, inputs, outputs) => {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  const offset = inputs[0].offset;
  const inBuf = inputs[0].storage.buffer as Float32Array;
  const outBuf = outputs[0].storage.buffer as Float32Array;
  for (let i = 0; i < outBuf.length; i++) {
    const v = inBuf[stridedIdx(shape, strides, offset, i)];
    outBuf[i] = v > 0 ? v : 0;
  }
};

// Backward: passes gradient where input was positive, zeros it elsewhere.
export const reluGradKernel: CPUKernel = (_node, inputs, outputs) => {
  // inputs[0] = grad, inputs[1] = original activation input a
  const shape = inputs[0].shape as number[];
  const outBuf = outputs[0].storage.buffer as Float32Array;
  const gradBuf = inputs[0].storage.buffer as Float32Array;
  const aBuf = inputs[1].storage.buffer as Float32Array;
  for (let i = 0; i < outBuf.length; i++) {
    const grad = gradBuf[stridedIdx(shape, inputs[0].strides, inputs[0].offset, i)];
    const a = aBuf[stridedIdx(shape, inputs[1].strides, inputs[1].offset, i)];
    outBuf[i] = a > 0 ? grad : 0;
  }
};
