import { Node } from '@minitensor/ir';
import { RuntimeTensor } from '@minitensor/runtime';
import { MinitensorWasmModule, WasmTensorHandle, isWasmTensorHandle, getF32View } from '../module';

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

/** Flat storage index for the element at logical position `flatIdx`. */
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
 * Returns true when the tensor is packed C-contiguous:
 * offset is 0 and strides are standard row-major values.
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

/**
 * Returns a WasmTensorHandle guaranteed to be packed contiguous.
 * If the input is already contiguous, returns its existing handle (owned=false).
 * Otherwise allocates a fresh WASM buffer, copies the data with strided reads,
 * and returns the new handle (owned=true). Caller must free owned handles with
 * module.free_f32(handle.ptr, handle.elements) once the kernel finishes.
 */
export function ensureContiguous(
  module: MinitensorWasmModule,
  tensor: RuntimeTensor,
): { handle: WasmTensorHandle; owned: boolean } {
  const shape = tensor.shape as number[];
  const handle = handleOf(tensor);
  if (isContiguous(shape, tensor.strides, tensor.offset)) {
    return { handle, owned: false };
  }

  // Non-contiguous: strided copy into a fresh WASM allocation.
  const total = getShapeSize(shape);
  const ptr = module.alloc_f32(total);
  const srcView = getF32View(module, handle);
  // Direct Float32Array view into the new WASM memory block
  const dstView = new Float32Array(module.memory.buffer, ptr, total);
  for (let i = 0; i < total; i++) {
    dstView[i] = srcView[stridedIdx(shape, tensor.strides, tensor.offset, i)];
  }
  return {
    handle: { ptr, elements: total, byteLength: total * Float32Array.BYTES_PER_ELEMENT },
    owned: true,
  };
}
