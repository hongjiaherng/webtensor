import { WASMKernel, handleOf } from '../../utils';

/**
 * Backward: passes the gradient through where the forward input was positive,
 * zeros it elsewhere. Always receives freshly-allocated contiguous tensors
 * from the autograd engine, so we use the `_raw` (no meta-buffer) variant.
 *
 * inputs[0] = grad, inputs[1] = original activation input `a`.
 */
export const reluBackwardKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const grad = handleOf(inputs[0]);
  const a = handleOf(inputs[1]);
  const out = handleOf(outputs[0]);
  module.relu_backward_raw(grad.ptr, a.ptr, out.ptr, out.elements);
};
