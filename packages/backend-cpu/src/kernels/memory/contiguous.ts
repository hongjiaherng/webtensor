import { CPUKernel, getShapeSize, stridedIdx, buf } from '../utils';

export const contiguousKernel: CPUKernel = (_node, inputs, outputs) => {
  const src = inputs[0];
  const srcBuf = buf(src);
  const dstBuf = buf(outputs[0]);
  const shape = src.shape as number[];
  const total = getShapeSize(shape);
  for (let i = 0; i < total; i++) {
    dstBuf[i] = srcBuf[stridedIdx(shape, src.strides, src.offset, i)];
  }
};
