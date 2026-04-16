import { Node, DType } from '@webtensor/ir';
import { Backend, RuntimeTensor, typedArrayCtor } from '@webtensor/runtime';
import { getShapeSize, computeContiguousStrides } from './kernels/utils';
import { cpuKernelRegistry } from './kernels/registry';

export class CPUBackend implements Backend {
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

  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void {
    const kernel = cpuKernelRegistry.get(node.op);
    if (!kernel) throw new Error('CPUBackend: unsupported op ' + node.op);
    kernel(node, inputs, outputs);
  }

  dispose(tensor: RuntimeTensor): void {
    if (tensor.isView) return;
    tensor.storage.buffer = null;
  }
}
