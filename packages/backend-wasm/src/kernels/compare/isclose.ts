import { WASMKernel, handleOf, allocMeta, buildBinaryMetaData } from '../utils';

/**
 * Element-wise `|a - b| <= atol + rtol * |b|`. Float32-only. Tolerances arrive
 * as node attributes; `equalNan === 1` turns NaN/NaN comparisons true.
 */
export const iscloseKernel: WASMKernel = (module, node, inputs, outputs) => {
  const rtol = node.attributes?.rtol as number;
  const atol = node.attributes?.atol as number;
  const equalNan = node.attributes?.equalNan as number;

  const meta = buildBinaryMetaData(inputs, outputs);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const b = handleOf(inputs[1]);
    const out = handleOf(outputs[0]);
    module.isclose_f32_strided(a.ptr, b.ptr, out.ptr, metaPtr, rtol, atol, equalNan);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
