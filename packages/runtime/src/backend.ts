import { Node, DType } from '@webtensor/ir';

/**
 * Physical memory backing a tensor. The runtime treats `buffer` as opaque —
 * only the backend that allocated it knows the concrete type and reads it.
 *
 * Concrete types per backend (each casts internally):
 *   @webtensor/backend-cpu    → TypedArray (Float32Array, Int32Array, Uint8Array)
 *   @webtensor/backend-wasm   → WasmTensorHandle ({ ptr, elements, byteLength })
 *   @webtensor/backend-webgpu → GPUBuffer
 *
 * Storage is separate from RuntimeTensor — the foundation for zero-copy
 * stride-based views.
 */
export interface RuntimeStorage {
  buffer: unknown;
  byteLength: number;
}

/**
 * A tensor as seen by the execution engine. Carries shape, dtype, strides and
 * an offset into the underlying storage.
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

/**
 * Device-level compute surface. Implementations are expected to expose an
 * async factory of the form `static async create(): Promise<Backend>` — see
 * the `BackendFactory` type in `./engine.ts`. TypeScript interfaces can't
 * express static methods, but the convention is uniform across all shipped
 * backends (`CPUBackend.create()`, `WASMBackend.create()`,
 * `WebGPUBackend.create()`) and is what `registerBackend()` consumes.
 */
export interface Backend {
  /**
   * Allocate fresh storage for a tensor of the given shape and dtype.
   * Synchronous — backends are expected to have allocation primitives that
   * don't require awaiting (ArrayBuffer ctor, GPUBuffer creation, WASM
   * `alloc_*`).
   */
  allocate(shape: (number | null)[], dtype: DType): RuntimeTensor;

  /**
   * Copy tensor data back to a JS TypedArray. This is the only genuinely
   * async call in the interface — WebGPU must round-trip through a staging
   * buffer (`mapAsync`), and CPU/WASM resolve immediately.
   */
  read(tensor: RuntimeTensor): Promise<ArrayBufferView>;

  /**
   * Write host-side `data` into the tensor's storage. Synchronous — on
   * WebGPU this is a `queue.writeBuffer` (queued, no await needed).
   */
  write(tensor: RuntimeTensor, data: ArrayBufferView): void;

  /**
   * Run one op. CPU/WASM compute in place and resolve immediately; WebGPU
   * submits commands to the GPU queue (which processes them in order) and
   * resolves without awaiting GPU completion — actual GPU work is awaited at
   * `read()` time via `mapAsync`, so per-op awaits would needlessly serialize
   * the pipeline.
   *
   * The Promise return type lets `Engine.evaluate` `await` uniformly across
   * backends, which also yields a microtask tick between ops and keeps the UI
   * responsive on large graphs.
   */
  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): Promise<void>;

  /** Free the tensor's storage. Synchronous. */
  dispose(tensor: RuntimeTensor): void;
}
