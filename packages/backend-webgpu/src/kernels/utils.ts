import { Node } from '@minitensor/ir';
import { RuntimeTensor } from '@minitensor/runtime';

export function getShapeSize(shape: (number | null)[]): number {
  let size = 1;
  for (const dim of shape) {
    if (dim === null) throw new Error("Dynamic dimensions not yet supported in WebGPU backend");
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

export function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

/**
 * Every op registers a WebGPUKernel. The backend calls these three methods and
 * knows nothing about op-specific shapes or uniforms.
 *
 * buildBindGroupEntries is responsible for creating any ephemeral uniform
 * buffers needed by the shader. Those buffers can be destroyed safely after
 * device.queue.submit() — WebGPU keeps the underlying resource alive until the
 * GPU is done with it.
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
// Shared helpers

/** Standard bind-group layout for ops that take N inputs + 1 output. */
export function elementwiseBindGroupEntries(
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[],
): GPUBindGroupEntry[] {
  return [
    ...inputs.map((t, i) => ({ binding: i, resource: { buffer: t.storage.buffer as GPUBuffer } })),
    { binding: inputs.length, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
  ];
}

/** 1-D workgroup dispatch: ceil(elements / 64) groups along X. */
export function flatDispatch(outputs: RuntimeTensor[]): [number, number, number] {
  const elements = getShapeSize(outputs[0].shape);
  return [Math.ceil(elements / 64), 1, 1];
}

/**
 * Create a 16-byte UNIFORM buffer pre-filled with the given u32 values.
 * Caller is responsible for destroying the buffer after submit().
 */
export function createUniformBuffer(device: GPUDevice, values: number[]): GPUBuffer {
  const buffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(values);
  buffer.unmap();
  return buffer;
}
