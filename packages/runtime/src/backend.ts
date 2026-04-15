import { Node } from '@minitensor/ir';

/**
 * Physical memory backing a tensor. Each backend stores its native handle here:
 *   CPU   — Float32Array
 *   WASM  — { ptr, elements, byteLength } (pointer into WASM linear memory)
 *   WebGPU — GPUBuffer
 *
 * Storage is never shared across tensors yet (no aliasing / views implemented),
 * but the separation from RuntimeTensor is the foundation for zero-copy views.
 */
export interface RuntimeStorage {
  buffer: any;       // Float32Array | GPUBuffer | WasmTensorHandle
  byteLength: number;
}

/**
 * A tensor as seen by the execution engine. Carries shape, dtype, strides and
 * an offset into the underlying storage — the same model as PyTorch's Tensor /
 * Storage split.
 *
 * Currently all allocated tensors are contiguous (offset = 0, C-order strides),
 * but the layout fields are present so ops like Transpose can later return
 * stride-based views without copying.
 */
export interface RuntimeTensor {
  storage: RuntimeStorage;
  shape: (number | null)[];
  strides: number[];
  offset: number;     // element offset into storage (0 for contiguous tensors)
  dtype: 'float32' | 'int32' | 'bool';
}

/**
 * Compute C-order (row-major) strides for the given concrete shape.
 * The last dimension has stride 1; each preceding dimension has stride equal
 * to the product of all dimensions after it.
 *
 * Example: shape [2, 3, 4] → strides [12, 4, 1]
 */
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
 * Returns true when the tensor occupies a contiguous region of storage in
 * C-order — i.e. offset is 0 and strides match computeContiguousStrides.
 * Non-contiguous tensors (e.g. transposed views) require a copy before being
 * passed to kernels that assume flat, packed buffers.
 */
export function isContiguous(tensor: RuntimeTensor): boolean {
  const shape = tensor.shape as number[];
  if (tensor.offset !== 0) return false;
  const expected = computeContiguousStrides(shape);
  return tensor.strides.every((s, i) => s === expected[i]);
}

export interface Backend {
  allocate(shape: (number | null)[], dtype: 'float32' | 'int32' | 'bool'): RuntimeTensor;
  read(tensor: RuntimeTensor): Promise<ArrayBufferView>;
  write(tensor: RuntimeTensor, data: ArrayBufferView): void;
  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void;
  dispose(tensor: RuntimeTensor): void;
}
