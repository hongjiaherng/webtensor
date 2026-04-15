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

export type CPUKernel = (node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]) => void;
