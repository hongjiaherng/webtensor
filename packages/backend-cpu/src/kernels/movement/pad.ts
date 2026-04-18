import { CPUKernel, computeContiguousStrides, getShapeSize, stridedIdx, buf } from '../utils';

/**
 * Constant-value Pad. Fills the output with `value`, then copies the input
 * into the `[pads_before, pads_before + input_shape)` region.
 */
export const padKernel: CPUKernel = (node, inputs, outputs) => {
  const pads = node.attributes!.pads as number[];
  const value = (node.attributes!.value as number) ?? 0;
  const outShape = outputs[0].shape as number[];
  const outStrides = computeContiguousStrides(outShape);
  const rank = outShape.length;
  const outBuf = buf(outputs[0]);

  // Fill first — cheaper than per-element branching.
  outBuf.fill(value);

  const src = inputs[0];
  const srcBuf = buf(src);
  const srcShape = src.shape as number[];
  const srcStrides = src.strides;
  const srcOffset = src.offset;
  const total = getShapeSize(srcShape);

  for (let flat = 0; flat < total; flat++) {
    let rem = flat;
    let outFlat = 0;
    for (let d = rank - 1; d >= 0; d--) {
      const coord = rem % srcShape[d];
      rem = Math.floor(rem / srcShape[d]);
      outFlat += (coord + pads[d]) * outStrides[d];
    }
    outBuf[outFlat] = srcBuf[stridedIdx(srcShape, srcStrides, srcOffset, flat)];
  }
};
