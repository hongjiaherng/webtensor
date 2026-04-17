// Bundler plugin (build.ts) transforms .wasm → base64 string default export.
// The wasm-pack-generated .d.ts shapes this as WASM exports, so cast to string.
import wasmBase64Raw from '../pkg/webtensor_wasm_bg.wasm';
const wasmBase64 = wasmBase64Raw as unknown as string;
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

type WasmExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  __wbindgen_externrefs: WebAssembly.Table;
  __wbindgen_start: () => void;
  alloc_f32: (len: number) => number;
  free_f32: (ptr: number, len: number) => void;
  alloc_u32: (len: number) => number;
  free_u32: (ptr: number, len: number) => void;
  add_strided: (a: number, b: number, c: number, d: number) => void;
  sub_strided: (a: number, b: number, c: number, d: number) => void;
  mul_strided: (a: number, b: number, c: number, d: number) => void;
  div_strided: (a: number, b: number, c: number, d: number) => void;
  relu_strided: (a: number, b: number, c: number) => void;
  relu_grad_raw: (a: number, b: number, c: number, d: number) => void;
  matmul_strided: (a: number, b: number, c: number, d: number) => void;
  neg_strided: (a: number, b: number, c: number) => void;
  exp_strided: (a: number, b: number, c: number) => void;
  log_strided: (a: number, b: number, c: number) => void;
  sqrt_strided: (a: number, b: number, c: number) => void;
  abs_strided: (a: number, b: number, c: number) => void;
  pow_strided: (a: number, b: number, c: number, d: number) => void;
  sigmoid_strided: (a: number, b: number, c: number) => void;
  tanh_strided: (a: number, b: number, c: number) => void;
};

let cachedModule: WebtensorWasmModule | null = null;

export async function loadWasmModule(): Promise<WebtensorWasmModule> {
  if (cachedModule) return cachedModule;

  const bytes = Uint8Array.from(atob(wasmBase64), (c) => c.charCodeAt(0));

  // The WASM binary imports one function: __wbindgen_init_externref_table.
  // Our kernels only use numeric pointers — no JS externrefs — so a no-op suffices.
  const { instance } = await WebAssembly.instantiate(bytes, {
    './webtensor_wasm_bg.js': {
      __wbindgen_init_externref_table: () => {},
    },
  });

  const x = instance.exports as WasmExports;
  x.__wbindgen_start();

  cachedModule = {
    memory: x.memory,
    alloc_f32: (len) => x.alloc_f32(len) >>> 0,
    free_f32: (ptr, len) => x.free_f32(ptr, len),
    alloc_u32: (len) => x.alloc_u32(len) >>> 0,
    free_u32: (ptr, len) => x.free_u32(ptr, len),
    add_strided: (a, b, c, d) => x.add_strided(a, b, c, d),
    sub_strided: (a, b, c, d) => x.sub_strided(a, b, c, d),
    mul_strided: (a, b, c, d) => x.mul_strided(a, b, c, d),
    div_strided: (a, b, c, d) => x.div_strided(a, b, c, d),
    relu_strided: (a, b, c) => x.relu_strided(a, b, c),
    relu_grad_raw: (a, b, c, d) => x.relu_grad_raw(a, b, c, d),
    matmul_strided: (a, b, c, d) => x.matmul_strided(a, b, c, d),
    neg_strided: (a, b, c) => x.neg_strided(a, b, c),
    exp_strided: (a, b, c) => x.exp_strided(a, b, c),
    log_strided: (a, b, c) => x.log_strided(a, b, c),
    sqrt_strided: (a, b, c) => x.sqrt_strided(a, b, c),
    abs_strided: (a, b, c) => x.abs_strided(a, b, c),
    pow_strided: (a, b, c, d) => x.pow_strided(a, b, c, d),
    sigmoid_strided: (a, b, c) => x.sigmoid_strided(a, b, c),
    tanh_strided: (a, b, c) => x.tanh_strided(a, b, c),
  };

  return cachedModule;
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
