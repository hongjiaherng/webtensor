import { CPUKernel, broadcastStridesOf, stridedIdx, buf } from '../utils';

type Pred = (a: number, b: number) => boolean;

/**
 * Factory for element-wise broadcast comparisons. All comparison kernels share
 * the same loop structure — they differ only in the predicate applied per
 * element. Input dtypes may be float32 or int32 (guaranteed same-dtype by the
 * core op guard). Output is always bool (Uint8Array, 0 / 1).
 */
export function compareKernel(pred: Pred): CPUKernel {
  return (_node, inputs, outputs) => {
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
      const av = aBuf[stridedIdx(outShape, aBcast, inputs[0].offset, i)];
      const bv = bBuf[stridedIdx(outShape, bBcast, inputs[1].offset, i)];
      outBuf[i] = pred(av, bv) ? 1 : 0;
    }
  };
}
