import { CPUKernel, buf, broadcastStridesOf } from '../utils';

// Batched matmul. Handles rank >= 2 with broadcast over leading batch dims.
export const matmulKernel: CPUKernel = (_node, inputs, outputs) => {
  const aShape = inputs[0].shape as number[];
  const bShape = inputs[1].shape as number[];
  const outShape = outputs[0].shape as number[];

  const rankA = aShape.length;
  const rankB = bShape.length;
  const outRank = outShape.length;

  const M = aShape[rankA - 2];
  const K = aShape[rankA - 1];
  const N = bShape[rankB - 1];

  const aRowStride = inputs[0].strides[rankA - 2];
  const aColStride = inputs[0].strides[rankA - 1];
  const bRowStride = inputs[1].strides[rankB - 2];
  const bColStride = inputs[1].strides[rankB - 1];
  const aOff = inputs[0].offset;
  const bOff = inputs[1].offset;

  const aBuf = buf(inputs[0]);
  const bBuf = buf(inputs[1]);
  const outBuf = buf(outputs[0]);

  // Batch shapes and strides
  const batchOutShape = outShape.slice(0, outRank - 2);
  const batchOutRank = batchOutShape.length;
  const aBatchShape = aShape.slice(0, rankA - 2);
  const bBatchShape = bShape.slice(0, rankB - 2);
  const aBatchStrides = inputs[0].strides.slice(0, rankA - 2);
  const bBatchStrides = inputs[1].strides.slice(0, rankB - 2);

  // Broadcast-align batch strides to batchOutShape
  const aBcast =
    batchOutRank === 0 ? [] : broadcastStridesOf(batchOutShape, aBatchShape, aBatchStrides);
  const bBcast =
    batchOutRank === 0 ? [] : broadcastStridesOf(batchOutShape, bBatchShape, bBatchStrides);

  const batchTotal = batchOutShape.reduce((acc, d) => acc * d, 1);
  const outMatStride = M * N; // outputs are contiguous

  const batchCoord = new Array<number>(batchOutRank).fill(0);

  for (let bIdx = 0; bIdx < batchTotal; bIdx++) {
    let rem = bIdx;
    for (let i = batchOutRank - 1; i >= 0; i--) {
      batchCoord[i] = rem % batchOutShape[i];
      rem = Math.floor(rem / batchOutShape[i]);
    }

    let aBatchOff = 0;
    let bBatchOff = 0;
    for (let i = 0; i < batchOutRank; i++) {
      aBatchOff += batchCoord[i] * aBcast[i];
      bBatchOff += batchCoord[i] * bBcast[i];
    }

    const outBase = bIdx * outMatStride;
    const aBase = aOff + aBatchOff;
    const bBase = bOff + bBatchOff;

    for (let row = 0; row < M; row++) {
      for (let col = 0; col < N; col++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum +=
            aBuf[aBase + row * aRowStride + k * aColStride] *
            bBuf[bBase + k * bRowStride + col * bColStride];
        }
        outBuf[outBase + row * N + col] = sum;
      }
    }
  }
};
