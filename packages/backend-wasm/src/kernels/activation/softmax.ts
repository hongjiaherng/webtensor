import { WASMKernel, handleOf, allocMeta, buildSoftmaxMetaData } from '../utils';

export const softmaxKernel: WASMKernel = (module, node, inputs, outputs) => {
  const axis = (node.attributes?.axis as number) ?? inputs[0].shape.length - 1;
  const meta = buildSoftmaxMetaData(inputs, axis);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const out = handleOf(outputs[0]);
    // Softmax is float-only (requires exp).
    module.softmax_f32_strided(a.ptr, out.ptr, metaPtr);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
