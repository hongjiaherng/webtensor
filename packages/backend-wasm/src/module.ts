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
  readonly alloc_i32: (len: number) => number;
  readonly free_i32: (ptr: number, len: number) => void;
  readonly alloc_u8: (len: number) => number;
  readonly free_u8: (ptr: number, len: number) => void;
  readonly alloc_u32: (len: number) => number;
  readonly free_u32: (ptr: number, len: number) => void;
  readonly add_f32_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly add_i32_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly sub_f32_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly sub_i32_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly mul_f32_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly mul_i32_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly div_f32_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly div_i32_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly relu_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly relu_backward_raw: (gradPtr: number, aPtr: number, outPtr: number, len: number) => void;
  readonly matmul_strided: (aPtr: number, bPtr: number, outPtr: number, metaPtr: number) => void;
  readonly reduce_sum_f32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly reduce_mean_f32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly reduce_all_u8_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly reduce_any_u8_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly softmax_f32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly neg_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly exp_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly log_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly sqrt_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly abs_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly pow_strided: (aPtr: number, outPtr: number, metaPtr: number, exponent: number) => void;
  readonly sigmoid_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly tanh_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly cast_f32_i32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly cast_f32_u8_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly cast_i32_f32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly cast_i32_u8_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly cast_u8_f32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly cast_u8_i32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly cast_f32_f32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly cast_i32_i32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly cast_u8_u8_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly eq_f32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly eq_i32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly ne_f32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly ne_i32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly lt_f32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly lt_i32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly le_f32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly le_i32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly gt_f32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly gt_i32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly ge_f32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly ge_i32_strided: (a: number, b: number, out: number, meta: number) => void;
  readonly isclose_f32_strided: (
    a: number,
    b: number,
    out: number,
    meta: number,
    rtol: number,
    atol: number,
    equalNan: number,
  ) => void;
  readonly concat_f32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly concat_i32_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly concat_u8_strided: (aPtr: number, outPtr: number, metaPtr: number) => void;
  readonly pad_f32_strided: (
    srcPtr: number,
    outPtr: number,
    metaPtr: number,
    value: number,
  ) => void;
  readonly pad_i32_strided: (
    srcPtr: number,
    outPtr: number,
    metaPtr: number,
    value: number,
  ) => void;
  readonly pad_u8_strided: (srcPtr: number, outPtr: number, metaPtr: number, value: number) => void;
}

type WasmExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  __wbindgen_externrefs: WebAssembly.Table;
  __wbindgen_start: () => void;
  alloc_f32: (len: number) => number;
  free_f32: (ptr: number, len: number) => void;
  alloc_i32: (len: number) => number;
  free_i32: (ptr: number, len: number) => void;
  alloc_u8: (len: number) => number;
  free_u8: (ptr: number, len: number) => void;
  alloc_u32: (len: number) => number;
  free_u32: (ptr: number, len: number) => void;
  add_f32_strided: (a: number, b: number, c: number, d: number) => void;
  add_i32_strided: (a: number, b: number, c: number, d: number) => void;
  sub_f32_strided: (a: number, b: number, c: number, d: number) => void;
  sub_i32_strided: (a: number, b: number, c: number, d: number) => void;
  mul_f32_strided: (a: number, b: number, c: number, d: number) => void;
  mul_i32_strided: (a: number, b: number, c: number, d: number) => void;
  div_f32_strided: (a: number, b: number, c: number, d: number) => void;
  div_i32_strided: (a: number, b: number, c: number, d: number) => void;
  relu_strided: (a: number, b: number, c: number) => void;
  relu_backward_raw: (a: number, b: number, c: number, d: number) => void;
  matmul_strided: (a: number, b: number, c: number, d: number) => void;
  reduce_sum_f32_strided: (a: number, b: number, c: number) => void;
  reduce_mean_f32_strided: (a: number, b: number, c: number) => void;
  reduce_all_u8_strided: (a: number, b: number, c: number) => void;
  reduce_any_u8_strided: (a: number, b: number, c: number) => void;
  softmax_f32_strided: (a: number, b: number, c: number) => void;
  neg_strided: (a: number, b: number, c: number) => void;
  exp_strided: (a: number, b: number, c: number) => void;
  log_strided: (a: number, b: number, c: number) => void;
  sqrt_strided: (a: number, b: number, c: number) => void;
  abs_strided: (a: number, b: number, c: number) => void;
  pow_strided: (a: number, b: number, c: number, d: number) => void;
  sigmoid_strided: (a: number, b: number, c: number) => void;
  tanh_strided: (a: number, b: number, c: number) => void;
  cast_f32_i32_strided: (a: number, b: number, c: number) => void;
  cast_f32_u8_strided: (a: number, b: number, c: number) => void;
  cast_i32_f32_strided: (a: number, b: number, c: number) => void;
  cast_i32_u8_strided: (a: number, b: number, c: number) => void;
  cast_u8_f32_strided: (a: number, b: number, c: number) => void;
  cast_u8_i32_strided: (a: number, b: number, c: number) => void;
  cast_f32_f32_strided: (a: number, b: number, c: number) => void;
  cast_i32_i32_strided: (a: number, b: number, c: number) => void;
  cast_u8_u8_strided: (a: number, b: number, c: number) => void;
  eq_f32_strided: (a: number, b: number, c: number, d: number) => void;
  eq_i32_strided: (a: number, b: number, c: number, d: number) => void;
  ne_f32_strided: (a: number, b: number, c: number, d: number) => void;
  ne_i32_strided: (a: number, b: number, c: number, d: number) => void;
  lt_f32_strided: (a: number, b: number, c: number, d: number) => void;
  lt_i32_strided: (a: number, b: number, c: number, d: number) => void;
  le_f32_strided: (a: number, b: number, c: number, d: number) => void;
  le_i32_strided: (a: number, b: number, c: number, d: number) => void;
  gt_f32_strided: (a: number, b: number, c: number, d: number) => void;
  gt_i32_strided: (a: number, b: number, c: number, d: number) => void;
  ge_f32_strided: (a: number, b: number, c: number, d: number) => void;
  ge_i32_strided: (a: number, b: number, c: number, d: number) => void;
  isclose_f32_strided: (
    a: number,
    b: number,
    c: number,
    d: number,
    rtol: number,
    atol: number,
    eqnan: number,
  ) => void;
  concat_f32_strided: (a: number, b: number, c: number) => void;
  concat_i32_strided: (a: number, b: number, c: number) => void;
  concat_u8_strided: (a: number, b: number, c: number) => void;
  pad_f32_strided: (a: number, b: number, c: number, v: number) => void;
  pad_i32_strided: (a: number, b: number, c: number, v: number) => void;
  pad_u8_strided: (a: number, b: number, c: number, v: number) => void;
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
    alloc_i32: (len) => x.alloc_i32(len) >>> 0,
    free_i32: (ptr, len) => x.free_i32(ptr, len),
    alloc_u8: (len) => x.alloc_u8(len) >>> 0,
    free_u8: (ptr, len) => x.free_u8(ptr, len),
    alloc_u32: (len) => x.alloc_u32(len) >>> 0,
    free_u32: (ptr, len) => x.free_u32(ptr, len),
    add_f32_strided: (a, b, c, d) => x.add_f32_strided(a, b, c, d),
    add_i32_strided: (a, b, c, d) => x.add_i32_strided(a, b, c, d),
    sub_f32_strided: (a, b, c, d) => x.sub_f32_strided(a, b, c, d),
    sub_i32_strided: (a, b, c, d) => x.sub_i32_strided(a, b, c, d),
    mul_f32_strided: (a, b, c, d) => x.mul_f32_strided(a, b, c, d),
    mul_i32_strided: (a, b, c, d) => x.mul_i32_strided(a, b, c, d),
    div_f32_strided: (a, b, c, d) => x.div_f32_strided(a, b, c, d),
    div_i32_strided: (a, b, c, d) => x.div_i32_strided(a, b, c, d),
    relu_strided: (a, b, c) => x.relu_strided(a, b, c),
    relu_backward_raw: (a, b, c, d) => x.relu_backward_raw(a, b, c, d),
    matmul_strided: (a, b, c, d) => x.matmul_strided(a, b, c, d),
    reduce_sum_f32_strided: (a, b, c) => x.reduce_sum_f32_strided(a, b, c),
    reduce_mean_f32_strided: (a, b, c) => x.reduce_mean_f32_strided(a, b, c),
    reduce_all_u8_strided: (a, b, c) => x.reduce_all_u8_strided(a, b, c),
    reduce_any_u8_strided: (a, b, c) => x.reduce_any_u8_strided(a, b, c),
    softmax_f32_strided: (a, b, c) => x.softmax_f32_strided(a, b, c),
    neg_strided: (a, b, c) => x.neg_strided(a, b, c),
    exp_strided: (a, b, c) => x.exp_strided(a, b, c),
    log_strided: (a, b, c) => x.log_strided(a, b, c),
    sqrt_strided: (a, b, c) => x.sqrt_strided(a, b, c),
    abs_strided: (a, b, c) => x.abs_strided(a, b, c),
    pow_strided: (a, b, c, d) => x.pow_strided(a, b, c, d),
    sigmoid_strided: (a, b, c) => x.sigmoid_strided(a, b, c),
    tanh_strided: (a, b, c) => x.tanh_strided(a, b, c),
    cast_f32_i32_strided: (a, b, c) => x.cast_f32_i32_strided(a, b, c),
    cast_f32_u8_strided: (a, b, c) => x.cast_f32_u8_strided(a, b, c),
    cast_i32_f32_strided: (a, b, c) => x.cast_i32_f32_strided(a, b, c),
    cast_i32_u8_strided: (a, b, c) => x.cast_i32_u8_strided(a, b, c),
    cast_u8_f32_strided: (a, b, c) => x.cast_u8_f32_strided(a, b, c),
    cast_u8_i32_strided: (a, b, c) => x.cast_u8_i32_strided(a, b, c),
    cast_f32_f32_strided: (a, b, c) => x.cast_f32_f32_strided(a, b, c),
    cast_i32_i32_strided: (a, b, c) => x.cast_i32_i32_strided(a, b, c),
    cast_u8_u8_strided: (a, b, c) => x.cast_u8_u8_strided(a, b, c),
    eq_f32_strided: (a, b, c, d) => x.eq_f32_strided(a, b, c, d),
    eq_i32_strided: (a, b, c, d) => x.eq_i32_strided(a, b, c, d),
    ne_f32_strided: (a, b, c, d) => x.ne_f32_strided(a, b, c, d),
    ne_i32_strided: (a, b, c, d) => x.ne_i32_strided(a, b, c, d),
    lt_f32_strided: (a, b, c, d) => x.lt_f32_strided(a, b, c, d),
    lt_i32_strided: (a, b, c, d) => x.lt_i32_strided(a, b, c, d),
    le_f32_strided: (a, b, c, d) => x.le_f32_strided(a, b, c, d),
    le_i32_strided: (a, b, c, d) => x.le_i32_strided(a, b, c, d),
    gt_f32_strided: (a, b, c, d) => x.gt_f32_strided(a, b, c, d),
    gt_i32_strided: (a, b, c, d) => x.gt_i32_strided(a, b, c, d),
    ge_f32_strided: (a, b, c, d) => x.ge_f32_strided(a, b, c, d),
    ge_i32_strided: (a, b, c, d) => x.ge_i32_strided(a, b, c, d),
    isclose_f32_strided: (a, b, c, d, rt, at, en) => x.isclose_f32_strided(a, b, c, d, rt, at, en),
    concat_f32_strided: (a, b, c) => x.concat_f32_strided(a, b, c),
    concat_i32_strided: (a, b, c) => x.concat_i32_strided(a, b, c),
    concat_u8_strided: (a, b, c) => x.concat_u8_strided(a, b, c),
    pad_f32_strided: (a, b, c, v) => x.pad_f32_strided(a, b, c, v),
    pad_i32_strided: (a, b, c, v) => x.pad_i32_strided(a, b, c, v),
    pad_u8_strided: (a, b, c, v) => x.pad_u8_strided(a, b, c, v),
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
