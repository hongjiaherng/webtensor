import { WASMKernel, handleOf, allocMeta, buildReduceMetaData } from '../utils';

export const anyKernel: WASMKernel = (module, node, inputs, outputs) => {
  const axes = (node.attributes?.axes as number[]) ?? [];
  const meta = buildReduceMetaData(inputs, axes);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const out = handleOf(outputs[0]);
    module.reduce_any_u8_strided(a.ptr, out.ptr, metaPtr);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
