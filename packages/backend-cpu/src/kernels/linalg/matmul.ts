import { CPUKernel } from '../utils';

export const matmulKernel: CPUKernel = (_node, inputs, outputs) => {
  const shapeA = inputs[0].shape as number[];
  const shapeB = inputs[1].shape as number[];
  const M = shapeA[shapeA.length - 2] ?? 1;
  const K = shapeA[shapeA.length - 1];
  const N = shapeB[shapeB.length - 1];

  const aStrides = inputs[0].strides;
  const bStrides = inputs[1].strides;
  const aOff = inputs[0].offset;
  const bOff = inputs[1].offset;
  // Last two strides: [..., rowStride, colStride]
  const aRowStride = aStrides[aStrides.length - 2] ?? K;
  const aColStride = aStrides[aStrides.length - 1];
  const bRowStride = bStrides[bStrides.length - 2] ?? N;
  const bColStride = bStrides[bStrides.length - 1];

  const aBuf = inputs[0].storage.buffer as Float32Array;
  const bBuf = inputs[1].storage.buffer as Float32Array;
  const outBuf = outputs[0].storage.buffer as Float32Array;

  for (let row = 0; row < M; row++) {
    for (let col = 0; col < N; col++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum +=
          aBuf[aOff + row * aRowStride + k * aColStride] *
          bBuf[bOff + k * bRowStride + col * bColStride];
      }
      outBuf[row * N + col] = sum;
    }
  }
};
