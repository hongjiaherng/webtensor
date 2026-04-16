import { CPUKernel, broadcastStridesOf, stridedIdx, buf } from '../utils';

export const divKernel: CPUKernel = (_node, inputs, outputs) => {
  const aShape = inputs[0].shape as number[];
  const bShape = inputs[1].shape as number[];
  const outShape = outputs[0].shape as number[];
  const aBuf = buf(inputs[0]);
  const bBuf = buf(inputs[1]);
  const outBuf = buf(outputs[0]);
  const total = outBuf.length;

  const aBcast = broadcastStridesOf(outShape, aShape, inputs[0].strides);
  const bBcast = broadcastStridesOf(outShape, bShape, inputs[1].strides);

  for (let i = 0; i < total; i++) {
    outBuf[i] =
      aBuf[stridedIdx(outShape, aBcast, inputs[0].offset, i)] /
      bBuf[stridedIdx(outShape, bBcast, inputs[1].offset, i)];
  }
};
