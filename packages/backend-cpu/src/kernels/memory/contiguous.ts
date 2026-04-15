import { CPUKernel, getShapeSize, stridedIdx } from '../utils';

export const contiguousKernel: CPUKernel = (_node, inputs, outputs) => {
  const src = inputs[0];
  const srcBuf = src.storage.buffer as Float32Array;
  const dstBuf = outputs[0].storage.buffer as Float32Array;
  const shape = src.shape as number[];
  const total = getShapeSize(shape);
  for (let i = 0; i < total; i++) {
    dstBuf[i] = srcBuf[stridedIdx(shape, src.strides, src.offset, i)];
  }
};
