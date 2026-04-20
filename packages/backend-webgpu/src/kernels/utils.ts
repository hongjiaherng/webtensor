import { Node, computeContiguousStrides, DType, MAX_RANK } from '@webtensor/ir';
import { RuntimeTensor, getShapeSize, broadcastStridesOf } from '@webtensor/runtime';

// ---------------------------------------------------------------------------
// Dtype → WGSL scalar type mapping.
// WGSL has no generics, so a kernel that supports multiple dtypes is compiled
// from a template: every `SCALAR` placeholder is replaced with `f32`/`i32`/
// `u32` at pipeline-build time. Bool values are packed as u32 (WebGPU storage
// buffers don't expose a 1-byte scalar; we use the low bit of u32).

export const WGSL_SCALAR: Record<DType, string> = {
  float32: 'f32',
  int32: 'i32',
  bool: 'u32',
};

// ---------------------------------------------------------------------------
// Shared WGSL struct for tensor metadata. Hoisted out of individual shaders so
// a layout change (e.g. lifting MAX_RANK) is one edit here, not 20 across the
// kernel tree. Injected into shaders via the `__TENSOR_META__` placeholder.
export const TENSOR_META_WGSL = `struct TensorMeta {
  rank:    u32,
  offset:  u32,
  shape:   array<u32, ${MAX_RANK}>,
  strides: array<u32, ${MAX_RANK}>,
};`;

/**
 * Inject the shared `TensorMeta` struct into a WGSL source at the
 * `__TENSOR_META__` placeholder. Safe to call on any shader; templates without
 * the placeholder are returned unchanged.
 */
export function injectMeta(template: string): string {
  return template.replace(/__TENSOR_META__/g, TENSOR_META_WGSL);
}

/** Substitute `SCALAR` and `__TENSOR_META__` in a WGSL template. */
export function renderWgsl(template: string, dtype: DType): string {
  return injectMeta(template).replace(/\bSCALAR\b/g, WGSL_SCALAR[dtype]);
}

export { computeContiguousStrides, getShapeSize, broadcastStridesOf };

// ---------------------------------------------------------------------------
// 1-D dispatch helper.
//
// WebGPU caps each dispatch dimension at `maxComputeWorkgroupsPerDimension`
// (65,535 by default). For 1-thread-per-output kernels with workgroup_size(64)
// that's ~4.19M elements in a single-axis dispatch — easily exceeded by a
// 2048² fp32 matrix. We spread the workgroups across X and Y.
//
// Shader-side contract: every kernel dispatched with `dispatch1D()` must
// declare a `num_workgroups` builtin and recover the flat index as
//   let i = gid.y * ng.x * WORKGROUP_SIZE_1D + gid.x;
// See e.g. add.wgsl for the canonical form.
export const WORKGROUP_SIZE_1D = 64;
const MAX_WORKGROUPS_PER_DIM = 65535;

export function dispatch1D(size: number): [number, number, number] {
  const totalGroups = Math.max(1, Math.ceil(size / WORKGROUP_SIZE_1D));
  if (totalGroups <= MAX_WORKGROUPS_PER_DIM) return [totalGroups, 1, 1];
  const y = Math.ceil(totalGroups / MAX_WORKGROUPS_PER_DIM);
  const x = Math.ceil(totalGroups / y);
  return [x, y, 1];
}

// ---------------------------------------------------------------------------
// TensorMeta uniform buffer
//
// WGSL struct layout (8 + 4*MAX_RANK*2 bytes, 2 + MAX_RANK*2 u32):
//   struct TensorMeta {
//     rank:    u32,
//     offset:  u32,
//     shape:   array<u32, MAX_RANK>,
//     strides: array<u32, MAX_RANK>,
//   }
//
// TypeScript packing:
//   [0]                           rank
//   [1]                           offset
//   [2 .. 2+MAX_RANK)             shape   (unused slots padded with 1)
//   [2+MAX_RANK .. 2+2*MAX_RANK)  strides (unused slots padded with 0)

const META_WORDS = 2 + MAX_RANK * 2;
const META_BYTES = META_WORDS * 4;
const SHAPE_OFFSET = 2;
const STRIDES_OFFSET = 2 + MAX_RANK;

/**
 * Pack a RuntimeTensor's metadata into a MAX_RANK-slot Uint32Array matching
 * the TensorMeta uniform struct layout. `outShape` (optional) overrides the
 * shape used for index decomposition — pass the broadcast output shape when
 * building meta for a binary-op input.
 *
 * When `outShape` is provided, strides are broadcast-adjusted: any dimension
 * where the tensor has size 1 but the output has size > 1 gets stride 0.
 */
export function packMeta(tensor: RuntimeTensor, outShape?: number[]): Uint32Array {
  const shape = (outShape ?? tensor.shape) as number[];
  const rank = shape.length;
  if (rank > MAX_RANK) {
    throw new Error(`packMeta: rank ${rank} exceeds MAX_RANK ${MAX_RANK}`);
  }
  const strides = outShape
    ? broadcastStridesOf(outShape, tensor.shape as number[], tensor.strides)
    : tensor.strides;

  const data = new Uint32Array(META_WORDS);
  data[0] = rank;
  data[1] = tensor.offset;
  for (let i = 0; i < MAX_RANK; i++) {
    data[SHAPE_OFFSET + i] = i < rank ? shape[i] : 1;
    data[STRIDES_OFFSET + i] = i < rank ? strides[i] : 0;
  }
  return data;
}

/**
 * Create a GPU uniform buffer pre-filled with TensorMeta data.
 */
export function createMetaBuffer(device: GPUDevice, data: Uint32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: META_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Create a GPU uniform buffer of arbitrary size (padded to 16 B alignment).
 * Used by op-specific auxiliary uniforms (reduce, softmax, batched matmul).
 */
export function createUniformBuffer(device: GPUDevice, data: Uint32Array): GPUBuffer {
  const paddedBytes = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
  const buffer = device.createBuffer({
    size: paddedBytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Pack batch-aware matmul meta for one input (A or B).
 * Produces a TensorMeta describing shape = [...batchOut, *matrixDims],
 * with strides broadcast-aligned on batch dims and preserving matrix strides.
 */
export function packMetaMatMulInput(tensor: RuntimeTensor, batchOutShape: number[]): Uint32Array {
  const shape = tensor.shape as number[];
  const rank = shape.length;
  const matrixRank = 2;
  const batchRank = batchOutShape.length;
  const outRank = batchRank + matrixRank;
  if (outRank > MAX_RANK) {
    throw new Error(`matmul: combined rank ${outRank} exceeds MAX_RANK ${MAX_RANK}`);
  }

  const batchShape = shape.slice(0, rank - matrixRank);
  const batchStrides = tensor.strides.slice(0, rank - matrixRank);
  const bcastBatch =
    batchRank === 0 ? [] : broadcastStridesOf(batchOutShape, batchShape, batchStrides);

  const data = new Uint32Array(META_WORDS);
  data[0] = outRank;
  data[1] = tensor.offset;
  for (let i = 0; i < MAX_RANK; i++) {
    if (i < batchRank) {
      data[SHAPE_OFFSET + i] = batchOutShape[i];
      data[STRIDES_OFFSET + i] = bcastBatch[i];
    } else if (i < outRank) {
      data[SHAPE_OFFSET + i] = shape[rank - matrixRank + (i - batchRank)];
      data[STRIDES_OFFSET + i] = tensor.strides[rank - matrixRank + (i - batchRank)];
    } else {
      data[SHAPE_OFFSET + i] = 1;
      data[STRIDES_OFFSET + i] = 0;
    }
  }
  return data;
}

/**
 * Pack BatchMeta uniform: (batch_rank, M, K, N, batch_out_shape[MAX_RANK]).
 * Matches WGSL layout of BatchMeta in matmul.wgsl.
 */
export function packBatchMeta(
  batchOutShape: number[],
  M: number,
  K: number,
  N: number,
): Uint32Array {
  const data = new Uint32Array(4 + MAX_RANK);
  data[0] = batchOutShape.length;
  data[1] = M;
  data[2] = K;
  data[3] = N;
  for (let i = 0; i < MAX_RANK; i++) {
    data[4 + i] = i < batchOutShape.length ? batchOutShape[i] : 1;
  }
  return data;
}

/**
 * Pack ReduceMeta uniform: (kept_rank, reduce_rank, kept_total, reduce_total,
 *                          kept_axes[MAX_RANK], reduce_axes[MAX_RANK]).
 */
export function packReduceMeta(
  keptAxes: number[],
  reduceAxes: number[],
  keptTotal: number,
  reduceTotal: number,
): Uint32Array {
  const data = new Uint32Array(4 + MAX_RANK * 2);
  data[0] = keptAxes.length;
  data[1] = reduceAxes.length;
  data[2] = keptTotal;
  data[3] = reduceTotal;
  for (let i = 0; i < MAX_RANK; i++) {
    data[4 + i] = i < keptAxes.length ? keptAxes[i] : 0;
    data[4 + MAX_RANK + i] = i < reduceAxes.length ? reduceAxes[i] : 0;
  }
  return data;
}

/**
 * Pack SoftmaxMeta uniform: (axis, slice_count, axis_len, _pad,
 *                            out_strides[MAX_RANK]).
 */
export function packSoftmaxMeta(
  axis: number,
  sliceCount: number,
  axisLen: number,
  outStrides: number[],
): Uint32Array {
  const data = new Uint32Array(4 + MAX_RANK);
  data[0] = axis;
  data[1] = sliceCount;
  data[2] = axisLen;
  for (let i = 0; i < MAX_RANK; i++) {
    data[4 + i] = i < outStrides.length ? outStrides[i] : 0;
  }
  return data;
}

// ---------------------------------------------------------------------------
// Kernel interface

/**
 * Every op registers a WebGPUKernel. The backend calls these three methods and
 * knows nothing about op-specific shapes, uniforms, or strides.
 *
 * buildBindGroupEntries creates any ephemeral meta buffers needed by the
 * shader and returns them in `tempBuffers` for cleanup after submission.
 */
export interface WebGPUKernel {
  /**
   * Build the compute pipeline. May vary per (node, inputs, outputs) — e.g.
   * dtype-aware kernels compile a different shader per dtype. The engine
   * caches pipelines by `${op}:${pipelineKey(...)}` if `pipelineKey` is set.
   */
  createPipeline(
    device: GPUDevice,
    node: Node,
    inputs: RuntimeTensor[],
    outputs: RuntimeTensor[],
  ): GPUComputePipeline;

  /**
   * Optional cache key distinguishing pipelines for this op. Return a short
   * string (e.g. dtype). Omit for ops that have a single pipeline.
   */
  pipelineKey?(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): string;

  buildBindGroupEntries(
    device: GPUDevice,
    node: Node,
    inputs: RuntimeTensor[],
    outputs: RuntimeTensor[],
  ): { entries: GPUBindGroupEntry[]; tempBuffers: GPUBuffer[] };

  getDispatch(
    node: Node,
    inputs: RuntimeTensor[],
    outputs: RuntimeTensor[],
  ): [number, number, number];

  /**
   * Optional escape hatch for kernels that need multiple dispatches (e.g.
   * Concat, which issues one dispatch per input). When present, the backend
   * skips its standard single-dispatch path and hands the op full control over
   * the encoder. The override must encode all compute passes it needs and
   * return any temp buffers for cleanup after submission.
   *
   * `buildBindGroupEntries` and `getDispatch` are not called when this is set.
   */
  executeOverride?(
    device: GPUDevice,
    encoder: GPUCommandEncoder,
    node: Node,
    inputs: RuntimeTensor[],
    outputs: RuntimeTensor[],
    pipeline: GPUComputePipeline,
  ): { tempBuffers: GPUBuffer[] };
}
