import { Node } from '@minitensor/ir';
import { Backend, RuntimeTensor } from '@minitensor/runtime';
import { getShapeSize } from './kernels/utils';
import { getF32View, isWasmTensorHandle, loadWasmModule, MinitensorWasmModule } from './module';
import { wasmKernelRegistry } from './kernels/registry';

export class WASMBackend implements Backend {
  private module: MinitensorWasmModule;

  private constructor(module: MinitensorWasmModule) {
    this.module = module;
  }

  static async create(): Promise<WASMBackend> {
    const module = await loadWasmModule();
    return new WASMBackend(module);
  }

  allocate(shape: (number | null)[], dtype: 'float32' | 'int32' | 'bool'): RuntimeTensor {
    if (dtype !== 'float32') {
      throw new Error('WASMBackend currently supports float32 tensors only');
    }
    const size = getShapeSize(shape);
    const ptr = this.module.alloc_f32(size);
    return {
      shape,
      dtype,
      buffer: {
        ptr,
        elements: size,
        byteLength: size * Float32Array.BYTES_PER_ELEMENT,
      },
    };
  }

  write(tensor: RuntimeTensor, data: ArrayBufferView): void {
    if (!isWasmTensorHandle(tensor.buffer)) throw new Error('WASMBackend: expected a WASM tensor handle');
    getF32View(this.module, tensor.buffer).set(data as Float32Array);
  }

  read(tensor: RuntimeTensor): Promise<ArrayBufferView> {
    if (!isWasmTensorHandle(tensor.buffer)) throw new Error('WASMBackend: expected a WASM tensor handle');
    return Promise.resolve(new Float32Array(getF32View(this.module, tensor.buffer)));
  }

  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void {
    const kernel = wasmKernelRegistry.get(node.op);
    if (!kernel) throw new Error('WASMBackend: unsupported op ' + node.op);
    kernel(this.module, node, inputs, outputs);
  }

  dispose(tensor: RuntimeTensor): void {
    if (isWasmTensorHandle(tensor.buffer)) {
      this.module.free_f32(tensor.buffer.ptr, tensor.buffer.elements);
    }
    tensor.buffer = null;
  }
}
