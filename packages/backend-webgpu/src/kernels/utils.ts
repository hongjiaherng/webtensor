import { Node } from '@minitensor/ir';
import {
  RuntimeTensor,
  computeContiguousStrides,
  broadcastStridesOf,
} from '@minitensor/runtime';

export { computeContiguousStrides, broadcastStridesOf };

export function getShapeSize(shape: (number | null)[]): number {
  let size = 1;
  for (const dim of shape) {
    if (dim === null) throw new Error('Dynamic dimensions not yet supported in WebGPU backend');
    size *= dim;
  }
  return size;
}

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
export function packMeta(
  tensor: RuntimeTensor,
  outShape?: number[],
): Uint32Array {
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
