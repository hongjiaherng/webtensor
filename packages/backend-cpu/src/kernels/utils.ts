import { Node } from '@minitensor/ir';
import {
  RuntimeTensor,
  computeContiguousStrides,
  stridedIdx,
  broadcastStridesOf,
  isContiguous,
} from '@minitensor/runtime';

export { computeContiguousStrides, stridedIdx, broadcastStridesOf, isContiguous };

export function getShapeSize(shape: (number | null)[]): number {
  let size = 1;
  for (const dim of shape) {
    if (dim === null) throw new Error('Dynamic dimensions not yet supported in CPU backend');
    size *= dim;
  }
  return size;
}

export type CPUKernel = (node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]) => void;
