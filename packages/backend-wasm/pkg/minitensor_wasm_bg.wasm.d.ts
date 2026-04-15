/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const add: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => void;
export const add_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
export const alloc_f32: (a: number) => number;
export const div: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => void;
export const div_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
export const free_f32: (a: number, b: number) => void;
export const matmul: (a: number, b: number, c: number, d: number, e: number, f: number, g: any, h: number, i: number, j: number) => void;
export const matmul_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
export const mul: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => void;
export const mul_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
export const relu: (a: number, b: number, c: number, d: number, e: any) => void;
export const relu_grad_raw: (a: number, b: number, c: number, d: number) => void;
export const relu_raw: (a: number, b: number, c: number) => void;
export const sub: (a: number, b: number, c: number, d: number, e: number, f: number, g: any) => void;
export const sub_raw: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
export const transpose: (a: number, b: number, c: number, d: number, e: any, f: number, g: number) => void;
export const transpose_raw: (a: number, b: number, c: number, d: number) => void;
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __wbindgen_start: () => void;
