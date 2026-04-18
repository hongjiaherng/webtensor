import { WASMKernel, handleOf, allocMeta, buildMatmulMetaData } from '../utils';

export const matmulKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const meta = buildMatmulMetaData(inputs, outputs);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const b = handleOf(inputs[1]);
    const out = handleOf(outputs[0]);
    module.matmul_strided(a.ptr, b.ptr, out.ptr, metaPtr);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
