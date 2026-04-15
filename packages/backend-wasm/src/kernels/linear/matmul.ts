import { WASMKernel, handleOf, ensureContiguous } from '../utils';

export const matmulKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const shapeA = inputs[0].shape as number[];
  const shapeB = inputs[1].shape as number[];
  const m = shapeA[shapeA.length - 2] || 1;
  const k = shapeA[shapeA.length - 1];
  const n = shapeB[shapeB.length - 1];
  const ca = ensureContiguous(module, inputs[0]);
  const cb = ensureContiguous(module, inputs[1]);
  try {
    module.matmul_raw(ca.handle.ptr, cb.handle.ptr, handleOf(outputs[0]).ptr, m, k, n);
  } finally {
    if (ca.owned) module.free_f32(ca.handle.ptr, ca.handle.elements);
    if (cb.owned) module.free_f32(cb.handle.ptr, cb.handle.elements);
  }
};
