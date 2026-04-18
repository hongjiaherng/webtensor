import { WASMKernel, handleOf, allocMeta, buildBinaryMetaData, dtypeSuffix } from '../utils';

/**
 * Factory for element-wise broadcast comparisons. Inputs must share one
 * arithmetic dtype (f32 or i32); output is u8. Dispatches to
 * `{opPrefix}_{f32|i32}_strided` based on `inputs[0].dtype`.
 */
export function compareKernel(opPrefix: 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge'): WASMKernel {
  return (module, _node, inputs, outputs) => {
    const meta = buildBinaryMetaData(inputs, outputs);
    const metaPtr = allocMeta(module, meta);
    try {
      const a = handleOf(inputs[0]);
      const b = handleOf(inputs[1]);
      const out = handleOf(outputs[0]);
      const fnName = `${opPrefix}_${dtypeSuffix(inputs[0].dtype)}_strided`;
      const fn = (module as unknown as Record<string, typeof module.eq_f32_strided>)[fnName];
      if (!fn) throw new Error(`WASM compare kernel '${fnName}' is not exported`);
      fn(a.ptr, b.ptr, out.ptr, metaPtr);
    } finally {
      module.free_u32(metaPtr, meta.length);
    }
  };
}
