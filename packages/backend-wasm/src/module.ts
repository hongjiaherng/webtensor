import initWasm, { InitOutput } from '../pkg/minitensor_wasm';

export interface WasmTensorHandle {
  ptr: number;
  elements: number;
  byteLength: number;
}

export interface MinitensorWasmModule extends InitOutput {
  readonly alloc_f32: (len: number) => number;
  readonly free_f32: (ptr: number, len: number) => void;
  readonly add_raw: (aPtr: number, bPtr: number, outPtr: number, lenA: number, lenB: number, lenOut: number) => void;
  readonly sub_raw: (aPtr: number, bPtr: number, outPtr: number, lenA: number, lenB: number, lenOut: number) => void;
  readonly mul_raw: (aPtr: number, bPtr: number, outPtr: number, lenA: number, lenB: number, lenOut: number) => void;
  readonly div_raw: (aPtr: number, bPtr: number, outPtr: number, lenA: number, lenB: number, lenOut: number) => void;
  readonly matmul_raw: (aPtr: number, bPtr: number, outPtr: number, m: number, k: number, n: number) => void;
  readonly transpose_raw: (aPtr: number, outPtr: number, m: number, n: number) => void;
  readonly relu_raw: (aPtr: number, outPtr: number, len: number) => void;
  readonly relu_grad_raw: (gradPtr: number, aPtr: number, outPtr: number, len: number) => void;
}

let wasmModule: MinitensorWasmModule | undefined;

export async function loadWasmModule(): Promise<MinitensorWasmModule> {
  if (!wasmModule) {
    wasmModule = await initWasm() as MinitensorWasmModule;
  }
  return wasmModule;
}

export function getF32View(module: MinitensorWasmModule, handle: WasmTensorHandle): Float32Array {
  return new Float32Array(module.memory.buffer, handle.ptr, handle.elements);
}

export function isWasmTensorHandle(value: unknown): value is WasmTensorHandle {
  return typeof value === 'object'
    && value !== null
    && typeof (value as WasmTensorHandle).ptr === 'number'
    && typeof (value as WasmTensorHandle).elements === 'number'
    && typeof (value as WasmTensorHandle).byteLength === 'number';
}
