import { WASMKernel, handleOf, allocMeta, buildBinaryMetaData } from '../utils';

export const mulKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const meta = buildBinaryMetaData(inputs, outputs);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const b = handleOf(inputs[1]);
    const out = handleOf(outputs[0]);
    module.mul_strided(a.ptr, b.ptr, out.ptr, metaPtr);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
