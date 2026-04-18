import {
  WASMKernel,
  handleOf,
  allocMeta,
  dtypeSuffix,
  buildConcatMetaData,
} from '../utils';

/**
 * Concat — scatter each input into the output along `axis`. The Rust kernel
 * handles one input at a time; we loop and advance `axisStart` between calls.
 */
export const concatKernel: WASMKernel = (module, node, inputs, outputs) => {
  const axis = node.attributes!.axis as number;
  const outShape = outputs[0].shape as number[];

  const suffix = dtypeSuffix(inputs[0].dtype);
  const fnName = `concat_${suffix}_strided`;
  const fn = (module as unknown as Record<string, typeof module.concat_f32_strided>)[fnName];
  if (!fn) throw new Error(`WASM concat kernel '${fnName}' is not exported`);

  const out = handleOf(outputs[0]);
  let axisStart = 0;
  for (const input of inputs) {
    const meta = buildConcatMetaData(input, outShape, axis, axisStart);
    const metaPtr = allocMeta(module, meta);
    try {
      const a = handleOf(input);
      fn(a.ptr, out.ptr, metaPtr);
    } finally {
      module.free_u32(metaPtr, meta.length);
    }
    axisStart += (input.shape as number[])[axis];
  }
};
