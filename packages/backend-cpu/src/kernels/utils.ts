import { Node, computeContiguousStrides } from '@webtensor/ir';
import {
  RuntimeTensor,
  TypedArray,
  getShapeSize,
  stridedIdx,
  broadcastStridesOf,
  isContiguous,
} from '@webtensor/runtime';

export { computeContiguousStrides, getShapeSize, stridedIdx, broadcastStridesOf, isContiguous };

/** Extract the typed buffer from a RuntimeTensor. */
export function buf(tensor: RuntimeTensor): TypedArray {
  return tensor.storage.buffer as TypedArray;
}

export type CPUKernel = (node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]) => void;
