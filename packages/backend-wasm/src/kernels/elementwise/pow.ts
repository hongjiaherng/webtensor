import { WASMKernel, handleOf, allocMeta, buildUnaryMetaData } from '../utils';

export const powKernel: WASMKernel = (module, node, inputs, outputs) => {
  const meta = buildUnaryMetaData(inputs);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const out = handleOf(outputs[0]);
    const exponent = node.attributes!.exponent as number;
    module.pow_strided(a.ptr, out.ptr, metaPtr, exponent);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
