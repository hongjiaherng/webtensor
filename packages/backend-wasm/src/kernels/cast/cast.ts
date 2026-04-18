import { WASMKernel, handleOf, allocMeta, buildUnaryMetaData, dtypeSuffix } from '../utils';

/**
 * Dtype conversion. Dispatches to one of 9 `cast_{from}_{to}_strided` kernels
 * based on the (input, output) dtype pair. Shares the 19-u32 unary meta layout.
 */
export const castKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const meta = buildUnaryMetaData(inputs);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const out = handleOf(outputs[0]);
    const fnName = `cast_${dtypeSuffix(inputs[0].dtype)}_${dtypeSuffix(outputs[0].dtype)}_strided`;
    const fn = (module as unknown as Record<string, typeof module.cast_f32_i32_strided>)[fnName];
    if (!fn) throw new Error(`WASM cast kernel '${fnName}' is not exported`);
    fn(a.ptr, out.ptr, metaPtr);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
