/* @ts-self-types="./minitensor_wasm.d.ts" */
import * as wasm from "./minitensor_wasm_bg.wasm";
import { __wbg_set_wasm } from "./minitensor_wasm_bg.js";

__wbg_set_wasm(wasm);
wasm.__wbindgen_start();
export {
    add, add_raw, alloc_f32, div, div_raw, free_f32, matmul, matmul_raw, mul, mul_raw, sub, sub_raw, transpose, transpose_raw
} from "./minitensor_wasm_bg.js";
