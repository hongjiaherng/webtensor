import { WASMKernel, handleOf, allocMeta, buildUnaryMetaData } from '../utils';

export const reluKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const meta = buildUnaryMetaData(inputs);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const out = handleOf(outputs[0]);
    module.relu_strided(a.ptr, out.ptr, metaPtr);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};

// reluGrad always receives freshly allocated contiguous tensors from the engine,
// so contiguous inputs are guaranteed — no strided handling needed here.
export const reluGradKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const grad = handleOf(inputs[0]);
  const a = handleOf(inputs[1]);
  const out = handleOf(outputs[0]);
  module.relu_grad_raw(grad.ptr, a.ptr, out.ptr, out.elements);
};
