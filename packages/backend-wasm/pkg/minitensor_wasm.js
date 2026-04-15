/* @ts-self-types="./minitensor_wasm.d.ts" */

/**
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {Float32Array} out
 */
export function add(a, b, out) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    var ptr2 = passArrayF32ToWasm0(out, wasm.__wbindgen_malloc);
    var len2 = WASM_VECTOR_LEN;
    wasm.add(ptr0, len0, ptr1, len1, ptr2, len2, out);
}

/**
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
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {Float32Array} out
 */
export function div(a, b, out) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    var ptr2 = passArrayF32ToWasm0(out, wasm.__wbindgen_malloc);
    var len2 = WASM_VECTOR_LEN;
    wasm.div(ptr0, len0, ptr1, len1, ptr2, len2, out);
}

/**
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
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {Float32Array} out
 * @param {number} m
 * @param {number} k
 * @param {number} n
 */
export function matmul(a, b, out, m, k, n) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    var ptr2 = passArrayF32ToWasm0(out, wasm.__wbindgen_malloc);
    var len2 = WASM_VECTOR_LEN;
    wasm.matmul(ptr0, len0, ptr1, len1, ptr2, len2, out, m, k, n);
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
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {Float32Array} out
 */
export function mul(a, b, out) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    var ptr2 = passArrayF32ToWasm0(out, wasm.__wbindgen_malloc);
    var len2 = WASM_VECTOR_LEN;
    wasm.mul(ptr0, len0, ptr1, len1, ptr2, len2, out);
}

/**
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
 * @param {Float32Array} a
 * @param {Float32Array} out
 */
export function relu(a, out) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    var ptr1 = passArrayF32ToWasm0(out, wasm.__wbindgen_malloc);
    var len1 = WASM_VECTOR_LEN;
    wasm.relu(ptr0, len0, ptr1, len1, out);
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
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {Float32Array} out
 */
export function sub(a, b, out) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(b, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    var ptr2 = passArrayF32ToWasm0(out, wasm.__wbindgen_malloc);
    var len2 = WASM_VECTOR_LEN;
    wasm.sub(ptr0, len0, ptr1, len1, ptr2, len2, out);
}

/**
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
 * @param {Float32Array} a
 * @param {Float32Array} out
 * @param {number} m
 * @param {number} n
 */
export function transpose(a, out, m, n) {
    const ptr0 = passArrayF32ToWasm0(a, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    var ptr1 = passArrayF32ToWasm0(out, wasm.__wbindgen_malloc);
    var len1 = WASM_VECTOR_LEN;
    wasm.transpose(ptr0, len0, ptr1, len1, out, m, n);
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
        __wbg___wbindgen_copy_to_typed_array_9e08990f20659111: function(arg0, arg1, arg2) {
            new Uint8Array(arg2.buffer, arg2.byteOffset, arg2.byteLength).set(getArrayU8FromWasm0(arg0, arg1));
        },
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

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
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
