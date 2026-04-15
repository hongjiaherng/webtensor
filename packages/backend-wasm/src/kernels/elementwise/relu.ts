import { WASMKernel, handleOf } from '../utils';

export const reluKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const a = handleOf(inputs[0]);
  const out = handleOf(outputs[0]);
  module.relu_raw(a.ptr, out.ptr, out.elements);
};

export const reluGradKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const grad = handleOf(inputs[0]);
  const a = handleOf(inputs[1]);
  const out = handleOf(outputs[0]);
  module.relu_grad_raw(grad.ptr, a.ptr, out.ptr, out.elements);
};
