import { CPUKernel, stridedIdx, buf } from '../utils';

export const powKernel: CPUKernel = (node, inputs, outputs) => {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  const offset = inputs[0].offset;
  const inBuf = buf(inputs[0]);
  const outBuf = buf(outputs[0]);
  const exponent = node.attributes!.exponent as number;
  for (let i = 0; i < outBuf.length; i++) {
    outBuf[i] = Math.pow(inBuf[stridedIdx(shape, strides, offset, i)], exponent);
  }
};
