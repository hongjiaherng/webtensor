import { WASMKernel, handleOf, ensureContiguous } from '../utils';

export const reluKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const ca = ensureContiguous(module, inputs[0]);
  try {
    const out = handleOf(outputs[0]);
    module.relu_raw(ca.handle.ptr, out.ptr, out.elements);
  } finally {
    if (ca.owned) module.free_f32(ca.handle.ptr, ca.handle.elements);
  }
};

export const reluGradKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const cGrad = ensureContiguous(module, inputs[0]);
  const cA = ensureContiguous(module, inputs[1]);
  try {
    const out = handleOf(outputs[0]);
    module.relu_grad_raw(cGrad.handle.ptr, cA.handle.ptr, out.ptr, out.elements);
  } finally {
    if (cGrad.owned) module.free_f32(cGrad.handle.ptr, cGrad.handle.elements);
    if (cA.owned) module.free_f32(cA.handle.ptr, cA.handle.elements);
  }
};
