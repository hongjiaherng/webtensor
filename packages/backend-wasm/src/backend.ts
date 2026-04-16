import { Node, DType } from '@webtensor/ir';
import { Backend, RuntimeTensor, TypedArray, bytesPerElement, copyBuffer, typedArrayCtor } from '@webtensor/runtime';
import { getShapeSize, computeContiguousStrides } from './kernels/utils';
import {
  getTypedView,
  isWasmTensorHandle,
  loadWasmModule,
  WebtensorWasmModule,
} from './module';
import { wasmKernelRegistry } from './kernels/registry';

export class WASMBackend implements Backend {
  private module: WebtensorWasmModule;

  private constructor(module: WebtensorWasmModule) {
    this.module = module;
  }

  static async create(): Promise<WASMBackend> {
    const module = await loadWasmModule();
    return new WASMBackend(module);
  }

  allocate(shape: (number | null)[], dtype: DType): RuntimeTensor {
    if (dtype !== 'float32') {
      throw new Error('WASMBackend currently supports float32 tensors only');
    }
    const size = getShapeSize(shape);
    const ptr = this.module.alloc_f32(size);
    const byteLength = size * bytesPerElement(dtype);
    return {
      storage: {
        buffer: { ptr, elements: size, byteLength },
        byteLength,
      },
      shape,
      strides: computeContiguousStrides(shape as number[]),
      offset: 0,
      dtype,
    };
  }

  write(tensor: RuntimeTensor, data: ArrayBufferView): void {
    if (!isWasmTensorHandle(tensor.storage.buffer))
      throw new Error('WASMBackend: expected a WASM tensor handle');
    const view = getTypedView(this.module, tensor.storage.buffer, tensor.dtype);
    copyBuffer(view, data);
  }

  read(tensor: RuntimeTensor): Promise<ArrayBufferView> {
    if (!isWasmTensorHandle(tensor.storage.buffer))
      throw new Error('WASMBackend: expected a WASM tensor handle');
    const handle = tensor.storage.buffer;
    // Copy out of WASM memory into a standalone JS TypedArray
    const Ctor = typedArrayCtor(tensor.dtype);
    const result = new Ctor(handle.elements);
    const view = getTypedView(this.module, handle, tensor.dtype);
    copyBuffer(result as TypedArray, view);
    return Promise.resolve(result as ArrayBufferView);
  }

  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void {
    const kernel = wasmKernelRegistry.get(node.op);
    if (!kernel) throw new Error('WASMBackend: unsupported op ' + node.op);
    kernel(this.module, node, inputs, outputs);
  }

  dispose(tensor: RuntimeTensor): void {
    if (tensor.isView) return;
    if (isWasmTensorHandle(tensor.storage.buffer)) {
      const handle = tensor.storage.buffer;
      this.module.free_f32(handle.ptr, handle.elements);
    }
    tensor.storage.buffer = null;
  }
}
