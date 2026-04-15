import { WASMKernel, handleOf, ensureContiguous } from '../utils';

export const subKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const ca = ensureContiguous(module, inputs[0]);
  const cb = ensureContiguous(module, inputs[1]);
  try {
    const out = handleOf(outputs[0]);
    module.sub_raw(ca.handle.ptr, cb.handle.ptr, out.ptr,
                   ca.handle.elements, cb.handle.elements, out.elements);
  } finally {
    if (ca.owned) module.free_f32(ca.handle.ptr, ca.handle.elements);
    if (cb.owned) module.free_f32(cb.handle.ptr, cb.handle.elements);
  }
};
