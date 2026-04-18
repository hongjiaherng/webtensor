import {
  WASMKernel,
  handleOf,
  allocMeta,
  dtypeSuffix,
  getShapeSize,
} from '../utils';

/**
 * Pad — fill output with `value`, scatter input into `[pads_before, pads_before + input_shape)`.
 * Meta layout matches `rust/src/ops/padding/pad.rs` (36 × u32).
 */
export const padKernel: WASMKernel = (module, node, inputs, outputs) => {
  const pads = node.attributes!.pads as number[];
  const value = (node.attributes!.value as number) ?? 0;
  const outShape = outputs[0].shape as number[];
  const rank = outShape.length;
  if (rank > 8) {
    throw new Error(`pad: rank ${rank} exceeds WASM kernel cap of 8`);
  }

  const suffix = dtypeSuffix(inputs[0].dtype);
  const fnName = `pad_${suffix}_strided`;
  const fn = (module as unknown as Record<string, typeof module.pad_f32_strided>)[fnName];
  if (!fn) throw new Error(`WASM pad kernel '${fnName}' is not exported`);

  const src = inputs[0];
  const srcShape = src.shape as number[];
  const srcStrides = src.strides;
  const total = getShapeSize(srcShape);

  const meta = new Uint32Array(36);
  meta[0] = total;
  meta[1] = rank;
  for (let i = 0; i < rank; i++) {
    meta[2 + i] = srcShape[i];
    meta[10 + i] = srcStrides[i];
    meta[19 + i] = outShape[i];
    meta[27 + i] = pads[i]; // pads_before
  }
  meta[18] = src.offset;

  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(src);
    const out = handleOf(outputs[0]);
    fn(a.ptr, out.ptr, metaPtr, value);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
