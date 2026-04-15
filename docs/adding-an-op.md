# Adding a New Op

Checklist for adding a kernel across all three backends. Follow in order — CPU establishes the correctness oracle before WASM or WebGPU are added.

---

## Step 1: Mark it in the roadmap

In `docs/next.md`, add a row to the **Ops** table (or relevant section) and update cells as each backend is completed.

---

## Step 2: Core graph-building function

**File:** `packages/core/src/ops.ts`

Add a function that constructs the IR node. Mirror the existing pattern:

```ts
export function sigmoid(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Sigmoid',
      inputs: [a],
      backward: (grad: Tensor) => {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        return [/* gradA */];
      },
    },
  });
}
```

The op string (`'Sigmoid'`) is what the kernel registries key on. It is already exported via `export * from './ops'` in `packages/core/src/index.ts`.

---

## Step 3: CPU kernel

**File:** `packages/backend-cpu/src/kernels/<category>/<OpName>.ts`

Categories: `elementwise/`, `linear/`, `shape/`, `reduction/`.

```ts
// packages/backend-cpu/src/kernels/elementwise/sigmoid.ts
import { CPUKernel } from '../utils';

export function executeSigmoid(a: Float32Array, out: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = 1 / (1 + Math.exp(-a[i]));
  }
}

export const sigmoidKernel: CPUKernel = (_node, inputs, outputs) => {
  executeSigmoid(inputs[0].buffer as Float32Array, outputs[0].buffer as Float32Array);
};
```

**Register in `packages/backend-cpu/src/kernels/registry.ts`:**

```ts
import { sigmoidKernel } from './elementwise/sigmoid';
// ...
['Sigmoid', sigmoidKernel],
```

---

## Step 4: CPU test

Add test cases in `tests/ops/<opname>.test.ts` using the `BACKENDS` array from `tests/helpers.ts`. Run against CPU first. Exclude backends that don't yet have the kernel by filtering:

```ts
BACKENDS.filter((b) => b.name !== 'WebGPU').forEach(({ name, create }) => { ... });
```

---

## Step 5: WASM kernel

**Rust function** in `packages/backend-wasm/rust/src/ops/<category>/<opname>.rs`:

```rust
use std::slice;

pub unsafe fn sigmoid_raw(a_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    let a = slice::from_raw_parts(a_ptr, len);
    let out = slice::from_raw_parts_mut(out_ptr, len);
    for i in 0..len {
        out[i] = 1.0 / (1.0 + (-a[i]).exp());
    }
}
```

Expose it via `wasm_bindgen` in `lib.rs`:

```rust
#[wasm_bindgen]
pub unsafe fn sigmoid_raw(a_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    ops::elementwise::sigmoid::sigmoid_raw(a_ptr, out_ptr, len);
}
```

**Extend `MinitensorWasmModule`** in `packages/backend-wasm/src/module.ts`:

```ts
readonly sigmoid_raw: (aPtr: number, outPtr: number, len: number) => void;
```

**TypeScript kernel** in `packages/backend-wasm/src/kernels/<category>/sigmoid.ts`:

```ts
import { WASMKernel, handleOf } from '../utils';

export const sigmoidKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const a = handleOf(inputs[0]);
  const out = handleOf(outputs[0]);
  module.sigmoid_raw(a.ptr, out.ptr, out.elements);
};
```

**Register in `packages/backend-wasm/src/kernels/registry.ts`.**

**Rebuild WASM** after changing Rust:

```sh
cd packages/backend-wasm && wasm-pack build rust --target bundler --out-dir ../pkg
```

---

## Step 6: WebGPU kernel

### WGSL shader

**File:** `packages/backend-webgpu/src/kernels/<category>/<OpName>.wgsl`

All kernels use the same `TensorMeta` uniform struct for strided access. Input tensors may have arbitrary strides and offsets (e.g. from a Transpose or Slice view) — the shader handles this transparently.

> **Important:** Do not name the uniform variable `meta` — it is a reserved keyword in WGSL. Use `u_meta` or another name.

**Unary op** (one input, one output):

```wgsl
struct TensorMeta {
  rank:    u32,
  offset:  u32,
  _p0:     u32,
  _p1:     u32,
  shape:   array<vec4<u32>, 2>,   // shape[0..7] packed as 2 × vec4
  strides: array<vec4<u32>, 2>,   // strides[0..7] packed as 2 × vec4
};

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read_write> out:    array<f32>;
@group(0) @binding(2) var<uniform>             u_meta: TensorMeta;

fn strided_idx(flat: u32) -> u32 {
  let rank = u_meta.rank;
  var rem = flat;
  var idx = u_meta.offset;
  for (var d = rank; d > 0u; d--) {
    let ax  = d - 1u;
    let dim = u_meta.shape[ax / 4u][ax % 4u];
    let s   = u_meta.strides[ax / 4u][ax % 4u];
    idx += (rem % dim) * s;
    rem  /= dim;
  }
  return idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out)) { return; }
  let x = a[strided_idx(i)];
  out[i] = 1.0 / (1.0 + exp(-x));  // sigmoid
}
```

**Binary op** (two inputs, one output): see `kernels/binary/add.wgsl` for the full pattern. Binary ops use two separate `TensorMeta` uniforms (`u_meta_a` at binding 3, `u_meta_b` at binding 4) and pass the broadcast output shape when building meta so that stride-0 broadcast dimensions are set correctly.

### TypeScript kernel

**File:** `packages/backend-webgpu/src/kernels/<category>/<OpName>.ts`

Implements the `WebGPUKernel` interface from `../utils`:

| Method | Responsibility |
| ------ | -------------- |
| `createPipeline(device)` | Compile the WGSL shader into a `GPUComputePipeline`. |
| `buildBindGroupEntries(device, node, inputs, outputs)` | Wire GPU buffers and meta uniform buffers to binding slots. Returns `{ entries, tempBuffers }` — `tempBuffers` are destroyed after GPU submission. |
| `getDispatch(node, inputs, outputs)` | Return `[x, y, z]` workgroup counts. |

**Unary example:**

```ts
import source from './sigmoid.wgsl?raw';
import { WebGPUKernel, packMeta, createMetaBuffer, getShapeSize } from '../utils';

export const sigmoidKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: source, label: 'SigmoidShader' }),
        entryPoint: 'main',
      },
      label: 'SigmoidPipeline',
    });
  },

  buildBindGroupEntries(device, _node, inputs, outputs) {
    const metaBuf = createMetaBuffer(device, packMeta(inputs[0]));
    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: metaBuf } },
      ],
      tempBuffers: [metaBuf],
    };
  },

  getDispatch(_node, _inputs, outputs) {
    return [Math.ceil(getShapeSize(outputs[0].shape) / 64), 1, 1];
  },
};
```

**Register in `packages/backend-webgpu/src/kernels/registry.ts`.**

No changes to `backend.ts` are needed — `execute()` is fully generic.

---

## Step 7: Parity test

Once all target backends support the op, add a `describe` block in `tests/backend/consistency.test.ts` that runs the same graph on each backend and asserts outputs match CPU within `1e-5`.
