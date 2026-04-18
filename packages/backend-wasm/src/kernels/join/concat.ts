import {
  WASMKernel,
  handleOf,
  allocMeta,
  dtypeSuffix,
  getShapeSize,
} from '../utils';

/**
 * Concat — scatter each input into the output along `axis`. The Rust kernel
 * handles one input at a time; we loop and advance `axisStart` between calls.
 * Meta layout matches `rust/src/ops/join/concat.rs` (29 × u32).
 */
export const concatKernel: WASMKernel = (module, node, inputs, outputs) => {
  const axis = node.attributes!.axis as number;
  const outShape = outputs[0].shape as number[];
  const rank = outShape.length;
  if (rank > 8) {
    throw new Error(`concat: rank ${rank} exceeds WASM kernel cap of 8`);
  }

  const suffix = dtypeSuffix(inputs[0].dtype);
  const fnName = `concat_${suffix}_strided`;
  const fn = (module as unknown as Record<string, typeof module.concat_f32_strided>)[fnName];
  if (!fn) throw new Error(`WASM concat kernel '${fnName}' is not exported`);

  const out = handleOf(outputs[0]);
  let axisStart = 0;
  for (const input of inputs) {
    const inShape = input.shape as number[];
    const inStrides = input.strides;
    const total = getShapeSize(inShape);

    const meta = new Uint32Array(29);
    meta[0] = total;
    meta[1] = rank;
    for (let i = 0; i < rank; i++) {
      meta[2 + i] = inShape[i];
      meta[10 + i] = inStrides[i];
      meta[19 + i] = outShape[i];
    }
    meta[18] = input.offset;
    meta[27] = axis;
    meta[28] = axisStart;

    const metaPtr = allocMeta(module, meta);
    try {
      const a = handleOf(input);
      fn(a.ptr, out.ptr, metaPtr);
    } finally {
      module.free_u32(metaPtr, meta.length);
    }
    axisStart += inShape[axis];
  }
};
