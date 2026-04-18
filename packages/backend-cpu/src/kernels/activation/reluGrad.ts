import { CPUKernel, stridedIdx, buf } from '../utils';

/**
 * Backward: passes the gradient through where the forward input was positive,
 * zeros it elsewhere.
 *
 * inputs[0] = grad, inputs[1] = original activation input `a`.
 */
export const reluGradKernel: CPUKernel = (_node, inputs, outputs) => {
  const shape = inputs[0].shape as number[];
  const outBuf = buf(outputs[0]);
  const gradBuf = buf(inputs[0]);
  const aBuf = buf(inputs[1]);
  for (let i = 0; i < outBuf.length; i++) {
    const grad = gradBuf[stridedIdx(shape, inputs[0].strides, inputs[0].offset, i)];
    const a = aBuf[stridedIdx(shape, inputs[1].strides, inputs[1].offset, i)];
    outBuf[i] = a > 0 ? grad : 0;
  }
};
