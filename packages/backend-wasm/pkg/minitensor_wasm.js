/* @ts-self-types="./minitensor_wasm.d.ts" */

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
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} meta_ptr
 */
export function add_strided(a_ptr, b_ptr, out_ptr, meta_ptr) {
    wasm.add_strided(a_ptr, b_ptr, out_ptr, meta_ptr);
}

/**
 * @param {number} len
 * @returns {number}
 */
export function alloc_f32(len) {
    const ret = wasm.alloc_f32(len);
    return ret >>> 0;
}

/**
 * Allocate a block of `len` u32 values for passing shape/stride meta buffers
 * from JavaScript into strided kernels.
 * @param {number} len
 * @returns {number}
 */
export function alloc_u32(len) {
    const ret = wasm.alloc_u32(len);
    return ret >>> 0;
}

/**
 * Strided element-wise divide.  Same meta layout as add_strided (28 × u32).
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} meta_ptr
 */
export function div_strided(a_ptr, b_ptr, out_ptr, meta_ptr) {
    wasm.div_strided(a_ptr, b_ptr, out_ptr, meta_ptr);
}

/**
 * @param {number} ptr
 * @param {number} len
 */
export function free_f32(ptr, len) {
    wasm.free_f32(ptr, len);
}

/**
 * @param {number} ptr
 * @param {number} len
 */
export function free_u32(ptr, len) {
    wasm.free_u32(ptr, len);
}

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
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} meta_ptr
 */
export function matmul_strided(a_ptr, b_ptr, out_ptr, meta_ptr) {
    wasm.matmul_strided(a_ptr, b_ptr, out_ptr, meta_ptr);
}

/**
 * Strided element-wise multiply.  Same meta layout as add_strided (28 × u32).
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} meta_ptr
 */
export function mul_strided(a_ptr, b_ptr, out_ptr, meta_ptr) {
    wasm.mul_strided(a_ptr, b_ptr, out_ptr, meta_ptr);
}

/**
 * Backward pass: passes gradient where the forward input was positive, zeros elsewhere.
 * Takes contiguous inputs (called from the autograd engine which always allocates fresh tensors).
 * @param {number} grad_ptr
 * @param {number} a_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
export function relu_grad_raw(grad_ptr, a_ptr, out_ptr, len) {
    wasm.relu_grad_raw(grad_ptr, a_ptr, out_ptr, len);
}

/**
 * Strided relu.
 *
 * meta layout (19 × u32):
 *   [0]      total elements
 *   [1]      rank
 *   [2..9]   shape[0..7]
 *   [10..17] strides[0..7]
 *   [18]     offset
 * @param {number} a_ptr
 * @param {number} out_ptr
 * @param {number} meta_ptr
 */
export function relu_strided(a_ptr, out_ptr, meta_ptr) {
    wasm.relu_strided(a_ptr, out_ptr, meta_ptr);
}

/**
 * Strided element-wise subtract.  Same meta layout as add_strided (28 × u32).
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} meta_ptr
 */
export function sub_strided(a_ptr, b_ptr, out_ptr, meta_ptr) {
    wasm.sub_strided(a_ptr, b_ptr, out_ptr, meta_ptr);
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./minitensor_wasm_bg.js": import0,
    };
}

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('minitensor_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
