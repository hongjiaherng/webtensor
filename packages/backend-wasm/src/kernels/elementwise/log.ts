import { WASMKernel, handleOf, allocMeta, buildUnaryMetaData } from '../utils';

export const logKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const meta = buildUnaryMetaData(inputs);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const out = handleOf(outputs[0]);
    module.log_strided(a.ptr, out.ptr, metaPtr);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
