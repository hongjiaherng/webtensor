import { WASMKernel, handleOf, allocMeta, buildBinaryMetaData, dtypeSuffix } from '../utils';

export const subKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const meta = buildBinaryMetaData(inputs, outputs);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const b = handleOf(inputs[1]);
    const out = handleOf(outputs[0]);
    const fnName = `sub_${dtypeSuffix(outputs[0].dtype)}_strided`;
    const fn = (module as unknown as Record<string, typeof module.sub_f32_strided>)[fnName];
    if (!fn) throw new Error(`sub: no WASM kernel for dtype ${outputs[0].dtype}`);
    fn(a.ptr, b.ptr, out.ptr, metaPtr);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
