import * as wasmExports from '../pkg/webtensor_wasm';
import { memory } from '../pkg/webtensor_wasm_bg.wasm';
import type { TypedArray } from '@webtensor/runtime';

export interface WasmTensorHandle {
  ptr: number;
  elements: number;
  byteLength: number;
}

export interface WebtensorWasmModule {
  readonly memory: WebAssembly.Memory;
  readonly alloc_f32: (len: number) => number;
  readonly free_f32: (ptr: number, len: number) => void;
  readonly alloc_u32: (len: number) => number;
  readonly free_u32: (ptr: number, len: number) => void;
  readonly add_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly sub_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly mul_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly div_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly relu_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly relu_grad_raw: (gradPtr: number, aPtr: number, outPtr: number, len: number) => void;
  readonly matmul_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly neg_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly exp_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly log_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly sqrt_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly abs_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly pow_strided: (aPtr: number, outPtr: number, metaPtr: number, exponent: number) => void;
  readonly sigmoid_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly tanh_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
}

const wasmModule: WebtensorWasmModule = {
  memory,
  ...wasmExports,
};

export async function loadWasmModule(): Promise<WebtensorWasmModule> {
  return wasmModule;
}

export function getF32View(module: WebtensorWasmModule, handle: WasmTensorHandle): Float32Array {
  return new Float32Array(module.memory.buffer, handle.ptr, handle.elements);
}

export function getTypedView(
  module: WebtensorWasmModule,
  handle: WasmTensorHandle,
  dtype: import('@webtensor/ir').DType,
): TypedArray {
  switch (dtype) {
    case 'float32':
      return new Float32Array(module.memory.buffer, handle.ptr, handle.elements);
    case 'int32':
      return new Int32Array(module.memory.buffer, handle.ptr, handle.elements);
    case 'bool':
      return new Uint8Array(module.memory.buffer, handle.ptr, handle.elements);
  }
}

export function isWasmTensorHandle(value: unknown): value is WasmTensorHandle {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as WasmTensorHandle).ptr === 'number' &&
    typeof (value as WasmTensorHandle).elements === 'number' &&
    typeof (value as WasmTensorHandle).byteLength === 'number'
  );
}
