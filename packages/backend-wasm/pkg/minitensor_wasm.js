/* @ts-self-types="./minitensor_wasm.d.ts" */

/**
 * Pointer-based entry point called from JavaScript.
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} len_a
 * @param {number} len_b
 * @param {number} len_out
 */
export function add_raw(a_ptr, b_ptr, out_ptr, len_a, len_b, len_out) {
    wasm.add_raw(a_ptr, b_ptr, out_ptr, len_a, len_b, len_out);
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
 * Pointer-based entry point called from JavaScript.
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} len_a
 * @param {number} len_b
 * @param {number} len_out
 */
export function div_raw(a_ptr, b_ptr, out_ptr, len_a, len_b, len_out) {
    wasm.div_raw(a_ptr, b_ptr, out_ptr, len_a, len_b, len_out);
}

/**
 * @param {number} ptr
 * @param {number} len
 */
export function free_f32(ptr, len) {
    wasm.free_f32(ptr, len);
}

/**
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} m
 * @param {number} k
 * @param {number} n
 */
export function matmul_raw(a_ptr, b_ptr, out_ptr, m, k, n) {
    wasm.matmul_raw(a_ptr, b_ptr, out_ptr, m, k, n);
}

/**
 * Pointer-based entry point called from JavaScript.
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} len_a
 * @param {number} len_b
 * @param {number} len_out
 */
export function mul_raw(a_ptr, b_ptr, out_ptr, len_a, len_b, len_out) {
    wasm.mul_raw(a_ptr, b_ptr, out_ptr, len_a, len_b, len_out);
}

/**
 * @param {number} grad_ptr
 * @param {number} a_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
export function relu_grad_raw(grad_ptr, a_ptr, out_ptr, len) {
    wasm.relu_grad_raw(grad_ptr, a_ptr, out_ptr, len);
}

/**
 * @param {number} a_ptr
 * @param {number} out_ptr
 * @param {number} len
 */
export function relu_raw(a_ptr, out_ptr, len) {
    wasm.relu_raw(a_ptr, out_ptr, len);
}

/**
 * Pointer-based entry point called from JavaScript.
 * @param {number} a_ptr
 * @param {number} b_ptr
 * @param {number} out_ptr
 * @param {number} len_a
 * @param {number} len_b
 * @param {number} len_out
 */
export function sub_raw(a_ptr, b_ptr, out_ptr, len_a, len_b, len_out) {
    wasm.sub_raw(a_ptr, b_ptr, out_ptr, len_a, len_b, len_out);
}

/**
 * @param {number} a_ptr
 * @param {number} out_ptr
 * @param {number} m
 * @param {number} n
 */
export function transpose_raw(a_ptr, out_ptr, m, n) {
    wasm.transpose_raw(a_ptr, out_ptr, m, n);
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
