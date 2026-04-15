import { Node } from '@minitensor/ir';
import { Backend, RuntimeTensor } from '@minitensor/runtime';
import { getShapeSize } from './kernels/utils';
import { cpuKernelRegistry } from './kernels/registry';

export class CPUBackend implements Backend {
  allocate(shape: (number | null)[], dtype: 'float32' | 'int32' | 'bool'): RuntimeTensor {
    if (dtype !== 'float32') {
      throw new Error(`CPUBackend: unsupported dtype '${dtype}' — only float32 is currently implemented`);
    }
    const size = getShapeSize(shape);
    return { shape, dtype, buffer: new Float32Array(size) };
  }

  read(tensor: RuntimeTensor): Promise<ArrayBufferView> {
    return Promise.resolve(tensor.buffer as ArrayBufferView);
  }

  write(tensor: RuntimeTensor, data: ArrayBufferView): void {
    const view = tensor.buffer as any;
    view.set(data as any);
  }

  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void {
    const kernel = cpuKernelRegistry.get(node.op);
    if (!kernel) throw new Error('CPUBackend: unsupported op ' + node.op);
    kernel(node, inputs, outputs);
  }

  dispose(tensor: RuntimeTensor): void {
    tensor.buffer = null;
  }
}
