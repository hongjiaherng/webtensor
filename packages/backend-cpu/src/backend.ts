import { Node } from '@minitensor/ir';
import { Backend, RuntimeTensor } from '@minitensor/runtime';
import { getShapeSize } from './kernels/utils';
import { cpuKernelRegistry } from './kernels/registry';

export class CPUBackend implements Backend {
  allocate(shape: (number | null)[], dtype: 'float32' | 'int32' | 'bool'): RuntimeTensor {
    const size = getShapeSize(shape);
    let buffer: ArrayBufferView;
    if (dtype === 'float32') {
      buffer = new Float32Array(size);
    } else if (dtype === 'int32') {
      buffer = new Int32Array(size);
    } else {
      buffer = new Uint8Array(size);
    }
    return { shape, dtype, buffer };
  }

  read(tensor: RuntimeTensor): ArrayBufferView {
    return tensor.buffer as ArrayBufferView;
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
