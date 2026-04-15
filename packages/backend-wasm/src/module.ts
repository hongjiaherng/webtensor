import initWasm, { InitOutput } from '../pkg/minitensor_wasm';

export interface WasmTensorHandle {
  ptr: number;
  elements: number;
  byteLength: number;
}

export interface MinitensorWasmModule extends InitOutput {
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
}

let wasmModule: MinitensorWasmModule | undefined;

export async function loadWasmModule(): Promise<MinitensorWasmModule> {
  if (!wasmModule) {
    wasmModule = (await initWasm()) as MinitensorWasmModule;
  }
  return wasmModule;
}

export function getF32View(module: MinitensorWasmModule, handle: WasmTensorHandle): Float32Array {
  return new Float32Array(module.memory.buffer, handle.ptr, handle.elements);
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
