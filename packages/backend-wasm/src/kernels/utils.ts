import { Node } from '@minitensor/ir';
import { RuntimeTensor } from '@minitensor/runtime';
import { MinitensorWasmModule, WasmTensorHandle, isWasmTensorHandle } from '../module';

export function getShapeSize(shape: (number | null)[]): number {
  let size = 1;
  for (const dim of shape) {
    if (dim === null) throw new Error('Dynamic dimensions not yet supported in WASM backend');
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

export type WASMKernel = (
  module: MinitensorWasmModule,
  node: Node,
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[],
) => void;

export function handleOf(tensor: RuntimeTensor): WasmTensorHandle {
  if (!isWasmTensorHandle(tensor.storage.buffer)) {
    throw new Error('WASMBackend: expected a WASM tensor handle');
  }
  return tensor.storage.buffer;
}
