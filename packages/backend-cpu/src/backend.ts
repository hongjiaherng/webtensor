import { Node } from '@minitensor/ir';
import { Backend, RuntimeTensor } from '@minitensor/runtime';
import { getShapeSize, computeContiguousStrides } from './kernels/utils';
import { cpuKernelRegistry } from './kernels/registry';

export class CPUBackend implements Backend {
  allocate(shape: (number | null)[], dtype: 'float32' | 'int32' | 'bool'): RuntimeTensor {
    if (dtype !== 'float32') {
      throw new Error(`CPUBackend: unsupported dtype '${dtype}' — only float32 is currently implemented`);
    }
    const size = getShapeSize(shape);
    const buffer = new Float32Array(size);
    return {
      storage: { buffer, byteLength: buffer.byteLength },
      shape,
      strides: computeContiguousStrides(shape as number[]),
      offset: 0,
      dtype,
    };
  }

  read(tensor: RuntimeTensor): Promise<ArrayBufferView> {
    return Promise.resolve(tensor.storage.buffer as Float32Array);
  }

  write(tensor: RuntimeTensor, data: ArrayBufferView): void {
    (tensor.storage.buffer as Float32Array).set(data as Float32Array);
  }

  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void {
    const kernel = cpuKernelRegistry.get(node.op);
    if (!kernel) throw new Error('CPUBackend: unsupported op ' + node.op);
    kernel(node, inputs, outputs);
  }

  dispose(tensor: RuntimeTensor): void {
    tensor.storage.buffer = null;
  }
}
