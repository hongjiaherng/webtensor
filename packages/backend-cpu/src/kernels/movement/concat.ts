import { CPUKernel, computeContiguousStrides, getShapeSize, stridedIdx, buf } from '../utils';

/**
 * Concat kernel — one `Concat` IR node takes N input tensors and produces one
 * contiguous output. Each input is scattered into the output along `axis` at
 * the running offset. Per-element read goes through strides so non-contiguous
 * inputs (views, post-transpose) are handled naturally.
 */
export const concatKernel: CPUKernel = (node, inputs, outputs) => {
  const axis = node.attributes!.axis as number;
  const outShape = outputs[0].shape as number[];
  const outStrides = computeContiguousStrides(outShape);
  const rank = outShape.length;
  const outBuf = buf(outputs[0]);

  let axisStart = 0;
  for (const input of inputs) {
    const inShape = input.shape as number[];
    const inBuf = buf(input);
    const inStrides = input.strides;
    const inOffset = input.offset;
    const total = getShapeSize(inShape);

    // For each flat index in the input, unravel its multi-index against the
    // input shape, shift the axis coord by `axisStart`, then re-ravel against
    // the contiguous output strides.
    for (let flat = 0; flat < total; flat++) {
      let rem = flat;
      let outFlat = 0;
      for (let d = rank - 1; d >= 0; d--) {
        const coord = rem % inShape[d];
        rem = Math.floor(rem / inShape[d]);
        const outCoord = d === axis ? coord + axisStart : coord;
        outFlat += outCoord * outStrides[d];
      }
      outBuf[outFlat] = inBuf[stridedIdx(inShape, inStrides, inOffset, flat)];
    }
    axisStart += inShape[axis];
  }
};
