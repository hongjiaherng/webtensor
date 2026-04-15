import { WASMKernel, handleOf } from '../utils';

export const mulKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const a = handleOf(inputs[0]);
  const b = handleOf(inputs[1]);
  const out = handleOf(outputs[0]);
  module.mul_raw(a.ptr, b.ptr, out.ptr, a.elements, b.elements, out.elements);
};
