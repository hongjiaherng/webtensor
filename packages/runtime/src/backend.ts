import { Node, DType } from '@webtensor/ir';

/**
 * Physical memory backing a tensor. Each backend stores its native handle here:
 *   CPU   — Float32Array
 *   WASM  — { ptr, elements, byteLength } (pointer into WASM linear memory)
 *   WebGPU — GPUBuffer
 *
 * Storage is separate from RuntimeTensor to mirror the PyTorch Storage/Tensor
 * split — the foundation for zero-copy stride-based views.
 */
export interface RuntimeStorage {
  buffer: any; // Float32Array | GPUBuffer | WasmTensorHandle
  byteLength: number;
}

/**
 * A tensor as seen by the execution engine. Carries shape, dtype, strides and
 * an offset into the underlying storage (same model as PyTorch's Tensor/Storage).
 *
 * All backends allocate with C-order strides and offset=0, but kernels must
 * handle arbitrary strides and offsets so that future zero-copy views (e.g.
 * a stride-based Transpose) work correctly without forced copies.
 */
export interface RuntimeTensor {
  storage: RuntimeStorage;
  shape: (number | null)[];
  strides: number[];
  offset: number; // element offset into storage (0 for contiguous tensors)
  dtype: DType;
  /**
   * True when this tensor is a zero-copy view of another tensor's storage.
   * The engine tracks the source tensor separately and ensures the source outlives
   * all its views. `backend.dispose()` must be a no-op for view tensors.
   */
  isView?: boolean;
}

// ---------------------------------------------------------------------------
// Stride utilities
// These live here — not in individual backends — so every backend imports
// from a single source of truth. Backends must not redefine these.

/**
 * Compute the total number of elements in a shape.
 */
export function getShapeSize(shape: (number | null)[]): number {
  let size = 1;
  for (const dim of shape) {
    size *= dim ?? 1;
  }
  return size;
}

/**
 * Compute the flat storage index for element at logical position `flatIdx`
 * inside a tensor with arbitrary `strides` and `offset`.
 *
 * Works from the innermost dimension outward, decomposing `flatIdx` into
 * per-axis coordinates and dotting with strides.
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
 * Compute the effective strides for an input tensor when broadcast to match
 * `outShape`. Axes that are broadcast — size 1 in the input, or missing (the
 * input has lower rank) — get stride 0 so repeated reads return the same
 * element without any data copying.
 */
export function broadcastStridesOf(
  outShape: number[],
  inShape: number[],
  inStrides: number[],
): number[] {
  const outRank = outShape.length;
  const result = new Array<number>(outRank).fill(0);
  const rankOffset = outRank - inShape.length; // right-align inShape with outShape
  for (let i = 0; i < inShape.length; i++) {
    result[rankOffset + i] = inShape[i] === 1 ? 0 : inStrides[i];
  }
  return result;
}

/**
 * Returns true when the tensor is packed C-contiguous: offset is 0 and strides
 * are exactly the row-major values for the given shape.
 *
 * Non-contiguous tensors (e.g. transposed views) require either a copy or
 * strided kernel support before further computation.
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

// ---------------------------------------------------------------------------

export interface Backend {
  allocate(shape: (number | null)[], dtype: DType): RuntimeTensor;
  read(tensor: RuntimeTensor): Promise<ArrayBufferView>;
  write(tensor: RuntimeTensor, data: ArrayBufferView): void;
  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void | Promise<void>;
  dispose(tensor: RuntimeTensor): void;
}
