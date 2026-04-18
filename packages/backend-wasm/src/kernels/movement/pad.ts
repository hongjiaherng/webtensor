import { WASMKernel, handleOf, allocMeta, dtypeSuffix, buildPadMetaData } from '../utils';

/**
 * Pad — fill output with `value`, scatter input into
 * `[pads_before, pads_before + input_shape)`.
 */
export const padKernel: WASMKernel = (module, node, inputs, outputs) => {
  const pads = node.attributes!.pads as number[];
  const value = (node.attributes!.value as number) ?? 0;
  const outShape = outputs[0].shape as number[];

  const suffix = dtypeSuffix(inputs[0].dtype);
  const fnName = `pad_${suffix}_strided`;
  const fn = (module as unknown as Record<string, typeof module.pad_f32_strided>)[fnName];
  if (!fn) throw new Error(`WASM pad kernel '${fnName}' is not exported`);

  const src = inputs[0];
  const meta = buildPadMetaData(src, outShape, pads);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(src);
    const out = handleOf(outputs[0]);
    fn(a.ptr, out.ptr, metaPtr, value);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
