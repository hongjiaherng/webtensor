import { Node, computeContiguousStrides } from '@webtensor/ir';
import { RuntimeTensor, getShapeSize, broadcastStridesOf } from '@webtensor/runtime';

export { computeContiguousStrides, getShapeSize, broadcastStridesOf };

// ---------------------------------------------------------------------------
// TensorMeta uniform buffer
//
// WGSL struct layout (80 bytes, 20 × u32):
//
//   struct TensorMeta {
//     rank:    u32,                     // bytes  0-3
//     offset:  u32,                     // bytes  4-7
//     _p0:     u32,                     // bytes  8-11  (padding)
//     _p1:     u32,                     // bytes 12-15  (padding)
//     shape:   array<vec4<u32>, 2>,     // bytes 16-47  shape[0..7]
//     strides: array<vec4<u32>, 2>,     // bytes 48-79  strides[0..7]
//   }
//
// Using var<uniform> (not storage) — uniform buffers with mappedAtCreation
// are proven reliable in Chromium's WebGPU implementation.
//
// TypeScript packing (u32 index → value):
//   [0]  rank
//   [1]  offset
//   [2]  0   (padding)
//   [3]  0   (padding)
//   [4..11]  shape[0..7]   (unused dims padded with 1)
//   [12..19] strides[0..7] (unused dims padded with 0)

const META_WORDS = 20; // 80 bytes
const META_BYTES = 80;

/**
 * Pack a RuntimeTensor's metadata into a 20-element Uint32Array that matches
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
  const strides = outShape
    ? broadcastStridesOf(outShape, tensor.shape as number[], tensor.strides)
    : tensor.strides;

  const data = new Uint32Array(META_WORDS);
  data[0] = rank;
  data[1] = tensor.offset;
  // data[2] and data[3] are padding zeros
  for (let i = 0; i < 8; i++) {
    data[4 + i] = i < rank ? shape[i] : 1;
    data[12 + i] = i < rank ? strides[i] : 0;
  }
  return data;
}

/**
 * Create a GPU uniform buffer pre-filled with TensorMeta data.
 * Uses mappedAtCreation for reliable initialization (proven to work for
 * uniform buffers in Chromium's WebGPU implementation).
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
 * Create a GPU uniform buffer of arbitrary size (multiple of 16 bytes).
 * Used by op-specific auxiliary uniforms (reduce, softmax, batched matmul).
 */
export function createUniformBuffer(device: GPUDevice, data: Uint32Array): GPUBuffer {
  const size = data.byteLength;
  const buffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Pack batch-aware matmul meta for one input (A or B).
 * Produces a 20-u32 TensorMeta describing shape = [...batchOut, *matrixDims],
 * with strides broadcast-aligned on batch dims and preserving matrix strides.
 */
export function packMetaMatMulInput(
  tensor: RuntimeTensor,
  batchOutShape: number[],
): Uint32Array {
  const shape = tensor.shape as number[];
  const rank = shape.length;
  const matrixRank = 2;
  const batchRank = batchOutShape.length;
  const outRank = batchRank + matrixRank;
  if (outRank > 8) {
    throw new Error(`matmul: combined rank ${outRank} exceeds WebGPU kernel cap of 8`);
  }

  const batchShape = shape.slice(0, rank - matrixRank);
  const batchStrides = tensor.strides.slice(0, rank - matrixRank);
  const bcastBatch =
    batchRank === 0 ? [] : broadcastStridesOf(batchOutShape, batchShape, batchStrides);

  const data = new Uint32Array(20);
  data[0] = outRank;
  data[1] = tensor.offset;
  for (let i = 0; i < 8; i++) {
    if (i < batchRank) {
      data[4 + i] = batchOutShape[i];
      data[12 + i] = bcastBatch[i];
    } else if (i < outRank) {
      data[4 + i] = shape[rank - matrixRank + (i - batchRank)];
      data[12 + i] = tensor.strides[rank - matrixRank + (i - batchRank)];
    } else {
      data[4 + i] = 1;
      data[12 + i] = 0;
    }
  }
  return data;
}

/**
 * Pack a 12-u32 BatchMeta uniform: (batch_rank, M, K, N, batch_out_shape[8]).
 * Matches WGSL layout of BatchMeta in matmul.wgsl.
 */
export function packBatchMeta(
  batchOutShape: number[],
  M: number,
  K: number,
  N: number,
): Uint32Array {
  const data = new Uint32Array(12);
  data[0] = batchOutShape.length;
  data[1] = M;
  data[2] = K;
  data[3] = N;
  for (let i = 0; i < 8; i++) {
    data[4 + i] = i < batchOutShape.length ? batchOutShape[i] : 1;
  }
  return data;
}

/**
 * Pack a 20-u32 ReduceMeta uniform:
 * (kept_rank, reduce_rank, kept_total, reduce_total, kept_axes[8], reduce_axes[8]).
 * Matches WGSL layout of ReduceMeta in reduce*.wgsl.
 */
export function packReduceMeta(
  keptAxes: number[],
  reduceAxes: number[],
  keptTotal: number,
  reduceTotal: number,
): Uint32Array {
  const data = new Uint32Array(20);
  data[0] = keptAxes.length;
  data[1] = reduceAxes.length;
  data[2] = keptTotal;
  data[3] = reduceTotal;
  for (let i = 0; i < 8; i++) {
    data[4 + i] = i < keptAxes.length ? keptAxes[i] : 0;
    data[12 + i] = i < reduceAxes.length ? reduceAxes[i] : 0;
  }
  return data;
}

/**
 * Pack a 12-u32 SoftmaxMeta uniform:
 * (axis, slice_count, axis_len, _pad, out_strides[8]).
 */
export function packSoftmaxMeta(
  axis: number,
  sliceCount: number,
  axisLen: number,
  outStrides: number[],
): Uint32Array {
  const data = new Uint32Array(12);
  data[0] = axis;
  data[1] = sliceCount;
  data[2] = axisLen;
  for (let i = 0; i < 8; i++) {
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
  createPipeline(device: GPUDevice): GPUComputePipeline;
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
}
