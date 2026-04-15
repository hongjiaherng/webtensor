/* tslint:disable */
/* eslint-disable */

export function add(a: Float32Array, b: Float32Array, out: Float32Array): void;

export function add_raw(a_ptr: number, b_ptr: number, out_ptr: number, len: number): void;

export function alloc_f32(len: number): number;

export function div(a: Float32Array, b: Float32Array, out: Float32Array): void;

export function div_raw(a_ptr: number, b_ptr: number, out_ptr: number, len: number): void;

export function free_f32(ptr: number, len: number): void;

export function matmul(a: Float32Array, b: Float32Array, out: Float32Array, m: number, k: number, n: number): void;

export function matmul_raw(a_ptr: number, b_ptr: number, out_ptr: number, m: number, k: number, n: number): void;

export function mul(a: Float32Array, b: Float32Array, out: Float32Array): void;

export function mul_raw(a_ptr: number, b_ptr: number, out_ptr: number, len: number): void;

export function sub(a: Float32Array, b: Float32Array, out: Float32Array): void;

export function sub_raw(a_ptr: number, b_ptr: number, out_ptr: number, len: number): void;

export function transpose(a: Float32Array, out: Float32Array, m: number, n: number): void;

export function transpose_raw(a_ptr: number, out_ptr: number, m: number, n: number): void;
