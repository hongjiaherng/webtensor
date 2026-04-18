import { CPUKernel, stridedIdx, buf } from '../utils';

/** Forward: `max(0, a)`. */
export const reluKernel: CPUKernel = (_node, inputs, outputs) => {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  const offset = inputs[0].offset;
  const inBuf = buf(inputs[0]);
  const outBuf = buf(outputs[0]);
  for (let i = 0; i < outBuf.length; i++) {
    const v = inBuf[stridedIdx(shape, strides, offset, i)];
    outBuf[i] = v > 0 ? v : 0;
  }
};
