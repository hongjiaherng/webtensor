/* tslint:disable */
/* eslint-disable */

export function add(a: Float32Array, b: Float32Array, out: Float32Array): void;

export function add_raw(a_ptr: number, b_ptr: number, out_ptr: number, len_a: number, len_b: number, len_out: number): void;

export function alloc_f32(len: number): number;

export function div(a: Float32Array, b: Float32Array, out: Float32Array): void;

export function div_raw(a_ptr: number, b_ptr: number, out_ptr: number, len_a: number, len_b: number, len_out: number): void;

export function free_f32(ptr: number, len: number): void;

export function matmul(a: Float32Array, b: Float32Array, out: Float32Array, m: number, k: number, n: number): void;

export function matmul_raw(a_ptr: number, b_ptr: number, out_ptr: number, m: number, k: number, n: number): void;

export function mul(a: Float32Array, b: Float32Array, out: Float32Array): void;

export function mul_raw(a_ptr: number, b_ptr: number, out_ptr: number, len_a: number, len_b: number, len_out: number): void;

export function relu(a: Float32Array, out: Float32Array): void;

export function relu_grad_raw(grad_ptr: number, a_ptr: number, out_ptr: number, len: number): void;

export function relu_raw(a_ptr: number, out_ptr: number, len: number): void;

export function sub(a: Float32Array, b: Float32Array, out: Float32Array): void;

export function sub_raw(a_ptr: number, b_ptr: number, out_ptr: number, len_a: number, len_b: number, len_out: number): void;

export function transpose(a: Float32Array, out: Float32Array, m: number, n: number): void;

export function transpose_raw(a_ptr: number, out_ptr: number, m: number, n: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly add: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => void;
    readonly add_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly alloc_f32: (a: number) => number;
    readonly div: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => void;
    readonly div_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly free_f32: (a: number, b: number) => void;
    readonly matmul: (a: number, b: number, c: number, d: number, e: number, f: number, g: any, h: number, i: number, j: number) => void;
    readonly matmul_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly mul: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => void;
    readonly mul_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly relu: (a: number, b: number, c: number, d: number, e: any) => void;
    readonly relu_grad_raw: (a: number, b: number, c: number, d: number) => void;
    readonly relu_raw: (a: number, b: number, c: number) => void;
    readonly sub: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => void;
    readonly sub_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly transpose: (a: number, b: number, c: number, d: number, e: any, f: number, g: number) => void;
    readonly transpose_raw: (a: number, b: number, c: number, d: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
