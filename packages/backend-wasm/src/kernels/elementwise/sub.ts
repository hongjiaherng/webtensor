import { WASMKernel, handleOf } from '../utils';

export const subKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const a = handleOf(inputs[0]);
  const b = handleOf(inputs[1]);
  const out = handleOf(outputs[0]);
  module.sub_raw(a.ptr, b.ptr, out.ptr, a.elements, b.elements, out.elements);
};
