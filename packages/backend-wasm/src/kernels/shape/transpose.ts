import { WASMKernel, handleOf, ensureContiguous } from '../utils';

export const transposeKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const shape = inputs[0].shape as number[];
  const m = shape[shape.length - 2] || 1;
  const n = shape[shape.length - 1];
  const ca = ensureContiguous(module, inputs[0]);
  try {
    module.transpose_raw(ca.handle.ptr, handleOf(outputs[0]).ptr, m, n);
  } finally {
    if (ca.owned) module.free_f32(ca.handle.ptr, ca.handle.elements);
  }
};
