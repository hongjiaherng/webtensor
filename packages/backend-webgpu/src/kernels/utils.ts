import { Node } from '@minitensor/ir';
import {
  RuntimeTensor,
  computeContiguousStrides,
  broadcastStridesOf,
  isContiguous,
} from '@minitensor/runtime';

export { computeContiguousStrides, broadcastStridesOf, isContiguous };

export function getShapeSize(shape: (number | null)[]): number {
  let size = 1;
  for (const dim of shape) {
    if (dim === null) throw new Error('Dynamic dimensions not yet supported in WebGPU backend');
    size *= dim;
  }
  return size;
}

export function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

/**
 * Every op registers a WebGPUKernel. The backend calls these three methods and
 * knows nothing about op-specific shapes, uniforms, or strides.
 *
 * buildBindGroupEntries is responsible for computing broadcast strides and
 * building any meta/uniform buffers the shader needs. Ephemeral buffers can
 * be destroyed safely after device.queue.submit() — WebGPU keeps the
 * underlying resource alive until the GPU is done with it.
 */
export interface WebGPUKernel {
  createPipeline(device: GPUDevice): GPUComputePipeline;
  buildBindGroupEntries(
    device: GPUDevice,
    node: Node,
    inputs: RuntimeTensor[],
    outputs: RuntimeTensor[],
  ): GPUBindGroupEntry[];
  getDispatch(
    node: Node,
    inputs: RuntimeTensor[],
    outputs: RuntimeTensor[],
  ): [number, number, number];
}

// ---------------------------------------------------------------------------
// Shared meta buffer helpers

/**
 * Build a storage-read buffer for binary elementwise shaders.
 * Layout (28 × u32 = 112 bytes):
 *   [0]      total elements in output
 *   [1]      rank
 *   [2..9]   out_shape[0..7]
 *   [10..17] a_broadcast_strides[0..7]
 *   [18]     a_offset
 *   [19..26] b_broadcast_strides[0..7]
 *   [27]     b_offset
 *
 * Uses storage (not uniform) to avoid WGSL's 16-byte element-padding rule
 * for array<u32, N> inside uniform structs.
 */
export function buildBinaryMeta(
  device: GPUDevice,
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[],
): GPUBuffer {
  const outShape = outputs[0].shape as number[];
  const aShape = inputs[0].shape as number[];
  const bShape = inputs[1].shape as number[];
  const total = getShapeSize(outShape);

  const aBcast = broadcastStridesOf(outShape, aShape, inputs[0].strides);
  const bBcast = broadcastStridesOf(outShape, bShape, inputs[1].strides);

  const data = new Uint32Array(28);
  data[0] = total;
  data[1] = outShape.length;
  for (let i = 0; i < outShape.length; i++) {
    data[2 + i] = outShape[i];
    data[10 + i] = aBcast[i];
    data[19 + i] = bBcast[i];
  }
  data[18] = inputs[0].offset;
  data[27] = inputs[1].offset;

  return createStorageBuffer(device, data);
}

/**
 * Build a storage-read buffer for unary elementwise shaders.
 * Layout (19 × u32 = 76 bytes):
 *   [0]      total elements
 *   [1]      rank
 *   [2..9]   shape[0..7]
 *   [10..17] strides[0..7]
 *   [18]     offset
 */
export function buildUnaryMeta(device: GPUDevice, inputs: RuntimeTensor[]): GPUBuffer {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  const total = getShapeSize(shape);

  const data = new Uint32Array(19);
  data[0] = total;
  data[1] = shape.length;
  for (let i = 0; i < shape.length; i++) {
    data[2 + i] = shape[i];
    data[10 + i] = strides[i];
  }
  data[18] = inputs[0].offset;

  return createStorageBuffer(device, data);
}

/** Create a GPU storage (read-only) buffer pre-filled with `data`. */
export function createStorageBuffer(device: GPUDevice, data: Uint32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: alignTo(data.byteLength, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Build a storage-read buffer for matmul shaders.
 * Layout (9 × u32 = 36 bytes):
 *   [0]  M
 *   [1]  K
 *   [2]  N
 *   [3]  a_row_stride  (A.strides[-2], or K if rank < 2)
 *   [4]  a_col_stride  (A.strides[-1])
 *   [5]  b_row_stride  (B.strides[-2], or N if rank < 2)
 *   [6]  b_col_stride  (B.strides[-1])
 *   [7]  a_offset
 *   [8]  b_offset
 */
export function buildMatmulMeta(device: GPUDevice, inputs: RuntimeTensor[]): GPUBuffer {
  const shapeA = inputs[0].shape as number[];
  const shapeB = inputs[1].shape as number[];
  const M = shapeA[shapeA.length - 2] ?? 1;
  const K = shapeA[shapeA.length - 1];
  const N = shapeB[shapeB.length - 1];

  const aStrides = inputs[0].strides;
  const bStrides = inputs[1].strides;
  const aRowStride = aStrides[aStrides.length - 2] ?? K;
  const aColStride = aStrides[aStrides.length - 1];
  const bRowStride = bStrides[bStrides.length - 2] ?? N;
  const bColStride = bStrides[bStrides.length - 1];

  const data = new Uint32Array(9);
  data[0] = M;
  data[1] = K;
  data[2] = N;
  data[3] = aRowStride;
  data[4] = aColStride;
  data[5] = bRowStride;
  data[6] = bColStride;
  data[7] = inputs[0].offset;
  data[8] = inputs[1].offset;

  return createStorageBuffer(device, data);
}

/**
 * Build a storage-read buffer for transpose shaders.
 * Layout (5 × u32 = 20 bytes):
 *   [0]  M            (rows of input)
 *   [1]  N            (cols of input)
 *   [2]  row_stride   (input.strides[-2])
 *   [3]  col_stride   (input.strides[-1])
 *   [4]  offset
 */
export function buildTransposeMeta(device: GPUDevice, inputs: RuntimeTensor[]): GPUBuffer {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  const M = shape[shape.length - 2] ?? 1;
  const N = shape[shape.length - 1];
  const rowStride = strides[strides.length - 2] ?? N;
  const colStride = strides[strides.length - 1];

  const data = new Uint32Array(5);
  data[0] = M;
  data[1] = N;
  data[2] = rowStride;
  data[3] = colStride;
  data[4] = inputs[0].offset;

  return createStorageBuffer(device, data);
}

/** 1-D workgroup dispatch: ceil(elements / 64) groups along X. */
export function flatDispatch(outputs: RuntimeTensor[]): [number, number, number] {
  const elements = getShapeSize(outputs[0].shape);
  return [Math.ceil(elements / 64), 1, 1];
}

// ---------------------------------------------------------------------------
// Legacy helpers (Option B — used by kernels before strided shaders)

/**
 * Simple bind group for binary elementwise ops: A, B, Out.
 * Relies on the gather pre-pass in WebGPUBackend to have made inputs contiguous.
 */
export function elementwiseBindGroupEntries(
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[],
): GPUBindGroupEntry[] {
  return [
    { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
    { binding: 1, resource: { buffer: inputs[1].storage.buffer as GPUBuffer } },
    { binding: 2, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
  ];
}

/**
 * Simple bind group for unary elementwise ops: A, Out.
 * Relies on the gather pre-pass in WebGPUBackend to have made inputs contiguous.
 */
export function unaryBindGroupEntries(
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[],
): GPUBindGroupEntry[] {
  return [
    { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
    { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
  ];
}

/** Create a uniform buffer pre-filled with 4 u32 values (16 bytes). */
export function createUniformBuffer(
  device: GPUDevice,
  values: [number, number, number, number],
): GPUBuffer {
  const buffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(values);
  buffer.unmap();
  return buffer;
}
