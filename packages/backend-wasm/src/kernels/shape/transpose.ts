import { WASMKernel, handleOf } from '../utils';

export const transposeKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const shape = inputs[0].shape as number[];
  const m = shape[shape.length - 2] || 1;
  const n = shape[shape.length - 1];
  module.transpose_raw(
    handleOf(inputs[0]).ptr,
    handleOf(outputs[0]).ptr,
    m, n,
  );
};
