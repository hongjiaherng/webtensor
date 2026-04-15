import { CPUKernel, broadcastStridesOf, stridedIdx } from '../utils';

export const mulKernel: CPUKernel = (_node, inputs, outputs) => {
  const aShape = inputs[0].shape as number[];
  const bShape = inputs[1].shape as number[];
  const outShape = outputs[0].shape as number[];
  const aBuf = inputs[0].storage.buffer as Float32Array;
  const bBuf = inputs[1].storage.buffer as Float32Array;
  const outBuf = outputs[0].storage.buffer as Float32Array;
  const total = outBuf.length;

  const aBcast = broadcastStridesOf(outShape, aShape, inputs[0].strides);
  const bBcast = broadcastStridesOf(outShape, bShape, inputs[1].strides);

  for (let i = 0; i < total; i++) {
    outBuf[i] = aBuf[stridedIdx(outShape, aBcast, inputs[0].offset, i)]
              * bBuf[stridedIdx(outShape, bBcast, inputs[1].offset, i)];
  }
};
