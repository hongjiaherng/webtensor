import { CPUKernel } from '../utils';

export const transposeKernel: CPUKernel = (_node, inputs, outputs) => {
  const shape = inputs[0].shape as number[];
  const M = shape[shape.length - 2] ?? 1;
  const N = shape[shape.length - 1];
  const strides = inputs[0].strides;
  const rowStride = strides[strides.length - 2] ?? N;
  const colStride = strides[strides.length - 1];
  const off = inputs[0].offset;
  const inBuf = inputs[0].storage.buffer as Float32Array;
  const outBuf = outputs[0].storage.buffer as Float32Array;

  for (let r = 0; r < M; r++) {
    for (let c = 0; c < N; c++) {
      outBuf[c * M + r] = inBuf[off + r * rowStride + c * colStride];
    }
  }
};
