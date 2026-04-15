/* tslint:disable */
/* eslint-disable */

/**
 * Strided element-wise add.
 *
 * meta layout (28 × u32):
 *   [0]      total elements in output
 *   [1]      rank
 *   [2..9]   out_shape[0..7]
 *   [10..17] a_broadcast_strides[0..7]
 *   [18]     a_offset
 *   [19..26] b_broadcast_strides[0..7]
 *   [27]     b_offset
 */
export function add_strided(a_ptr: number, b_ptr: number, out_ptr: number, meta_ptr: number): void;

export function alloc_f32(len: number): number;

/**
 * Allocate a block of `len` u32 values for passing shape/stride meta buffers
 * from JavaScript into strided kernels.
 */
export function alloc_u32(len: number): number;

/**
 * Strided element-wise divide.  Same meta layout as add_strided (28 × u32).
 */
export function div_strided(a_ptr: number, b_ptr: number, out_ptr: number, meta_ptr: number): void;

export function free_f32(ptr: number, len: number): void;

export function free_u32(ptr: number, len: number): void;

/**
 * Strided 2-D matrix multiply.
 *
 * meta layout (9 × u32):
 *   [0]  M
 *   [1]  K
 *   [2]  N
 *   [3]  a_row_stride   (A.strides[rank-2])
 *   [4]  a_col_stride   (A.strides[rank-1])
 *   [5]  b_row_stride
 *   [6]  b_col_stride
 *   [7]  a_offset
 *   [8]  b_offset
 */
export function matmul_strided(a_ptr: number, b_ptr: number, out_ptr: number, meta_ptr: number): void;

/**
 * Strided element-wise multiply.  Same meta layout as add_strided (28 × u32).
 */
export function mul_strided(a_ptr: number, b_ptr: number, out_ptr: number, meta_ptr: number): void;

/**
 * Backward pass: passes gradient where the forward input was positive, zeros elsewhere.
 * Takes contiguous inputs (called from the autograd engine which always allocates fresh tensors).
 */
export function relu_grad_raw(grad_ptr: number, a_ptr: number, out_ptr: number, len: number): void;

/**
 * Strided relu.
 *
 * meta layout (19 × u32):
 *   [0]      total elements
 *   [1]      rank
 *   [2..9]   shape[0..7]
 *   [10..17] strides[0..7]
 *   [18]     offset
 */
export function relu_strided(a_ptr: number, out_ptr: number, meta_ptr: number): void;

/**
 * Strided element-wise subtract.  Same meta layout as add_strided (28 × u32).
 */
export function sub_strided(a_ptr: number, b_ptr: number, out_ptr: number, meta_ptr: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly add_strided: (a: number, b: number, c: number, d: number) => void;
    readonly alloc_f32: (a: number) => number;
    readonly div_strided: (a: number, b: number, c: number, d: number) => void;
    readonly free_f32: (a: number, b: number) => void;
    readonly matmul_strided: (a: number, b: number, c: number, d: number) => void;
    readonly mul_strided: (a: number, b: number, c: number, d: number) => void;
    readonly relu_grad_raw: (a: number, b: number, c: number, d: number) => void;
    readonly relu_strided: (a: number, b: number, c: number) => void;
    readonly sub_strided: (a: number, b: number, c: number, d: number) => void;
    readonly free_u32: (a: number, b: number) => void;
    readonly alloc_u32: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
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
