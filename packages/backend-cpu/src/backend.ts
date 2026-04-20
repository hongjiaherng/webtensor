import { Node, DType } from '@webtensor/ir';
import { Backend, RuntimeTensor, typedArrayCtor } from '@webtensor/runtime';
import { getShapeSize, computeContiguousStrides } from './kernels/utils';
import { cpuKernelRegistry } from './kernels/registry';

export class CPUBackend implements Backend {
  private constructor() {}

  /**
   * Async factory to match the `Backend` lifecycle convention used by WASM
   * and WebGPU. CPU has no setup work today, but keeping the shape uniform
   * lets consumer code stay backend-agnostic (`await Backend.create()` works
   * everywhere) and leaves room for future async init (SIMD polyfill, etc.)
   * without a breaking API change.
   */
  static async create(): Promise<CPUBackend> {
    return new CPUBackend();
  }

  allocate(shape: (number | null)[], dtype: DType): RuntimeTensor {
    const size = getShapeSize(shape);
    const Ctor = typedArrayCtor(dtype);
    const buffer = new Ctor(size);
    return {
      storage: { buffer, byteLength: buffer.byteLength },
      shape,
      strides: computeContiguousStrides(shape as number[]),
      offset: 0,
      dtype,
    };
  }

  read(tensor: RuntimeTensor): Promise<ArrayBufferView> {
    return Promise.resolve(tensor.storage.buffer as ArrayBufferView);
  }

  write(tensor: RuntimeTensor, data: ArrayBufferView): void {
    const buf = tensor.storage.buffer as { set(src: ArrayBufferView): void };
    buf.set(data);
  }

  async execute(
    node: Node,
    inputs: RuntimeTensor[],
    outputs: RuntimeTensor[],
  ): Promise<void> {
    const kernel = cpuKernelRegistry.get(node.op);
    if (!kernel) throw new Error('CPUBackend: unsupported op ' + node.op);
    kernel(node, inputs, outputs);
  }

  dispose(tensor: RuntimeTensor): void {
    if (tensor.isView) return;
    tensor.storage.buffer = null;
  }
}
