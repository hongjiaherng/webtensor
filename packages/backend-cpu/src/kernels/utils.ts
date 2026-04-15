import { Node } from '@minitensor/ir';
import { RuntimeTensor } from '@minitensor/runtime';

export function getShapeSize(shape: (number | null)[]): number {
  let size = 1;
  for (const dim of shape) {
    if (dim === null) throw new Error('Dynamic dimensions not yet supported in CPU backend');
    size *= dim;
  }
  return size;
}

/** C-order (row-major) strides for a concrete shape. */
export function computeContiguousStrides(shape: number[]): number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Flat storage index for the element at logical position `flatIdx` in a tensor
 * with arbitrary strides and offset. Works from the innermost dimension outward,
 * decomposing flatIdx into per-axis coordinates then dotting with strides.
 */
export function stridedIdx(
  shape: number[],
  strides: number[],
  offset: number,
  flatIdx: number,
): number {
  let remaining = flatIdx;
  let idx = offset;
  for (let i = shape.length - 1; i >= 0; i--) {
    idx += (remaining % shape[i]) * strides[i];
    remaining = Math.floor(remaining / shape[i]);
  }
  return idx;
}

/**
 * Broadcast strides for an input tensor when it is logically expanded to match
 * `outShape`. Any axis that is broadcast (size 1 in input, or the axis does not
 * exist in input) gets stride 0, so repeated reads return the same element.
 */
export function broadcastStridesOf(
  outShape: number[],
  inShape: number[],
  inStrides: number[],
): number[] {
  const outRank = outShape.length;
  const result = new Array<number>(outRank).fill(0);
  const offset = outRank - inShape.length; // right-align inShape with outShape
  for (let i = 0; i < inShape.length; i++) {
    result[offset + i] = inShape[i] === 1 ? 0 : inStrides[i];
  }
  return result;
}

/**
 * Returns true when the tensor data is packed C-contiguous:
 * offset is 0 and strides are the standard row-major values.
 */
export function isContiguous(shape: number[], strides: number[], offset: number): boolean {
  if (offset !== 0) return false;
  let expected = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    if (strides[i] !== expected) return false;
    expected *= shape[i];
  }
  return true;
}

export type CPUKernel = (node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]) => void;
