import { WASMKernel, handleOf } from '../utils';

export const matmulKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const shapeA = inputs[0].shape as number[];
  const shapeB = inputs[1].shape as number[];
  const m = shapeA[shapeA.length - 2] || 1;
  const k = shapeA[shapeA.length - 1];
  const n = shapeB[shapeB.length - 1];
  module.matmul_raw(
    handleOf(inputs[0]).ptr,
    handleOf(inputs[1]).ptr,
    handleOf(outputs[0]).ptr,
    m, k, n,
  );
};
