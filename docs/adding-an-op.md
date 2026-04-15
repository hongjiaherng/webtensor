# Adding a New Op

Checklist for adding a kernel across all three backends. Follow in order — CPU establishes the correctness oracle before WASM or WebGPU are added.

---

## Step 1: Update the kernel support matrix

In `README.md`, add a row to the **Kernel Support Matrix** table with `—` for all backends. Update cells to `yes` as each backend is completed.

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
        // Requires a SigmoidGrad op or reuses forward output — add later
        return [/* gradA */];
      },
    },
  });
}
```

The op string (`'Sigmoid'`) is what the kernel registries key on. Export it from `packages/core/src/index.ts` if it is not already covered by `export * from './ops'`.

---

## Step 3: CPU kernel

**File:** `packages/backend-cpu/src/kernels/<category>/<OpName>.ts`

Categories: `elementwise/`, `linear/`, `shape/`, `reduction/` (create a new one if none fit).

Each kernel file exports two things:

1. A pure `executeXxx` function that operates on typed arrays — easy to unit-test in isolation.
2. A named `xxxKernel: CPUKernel` that bridges the runtime tensor API to the typed-array function.

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

**Binary elementwise** — same pattern, two inputs:

```ts
export function executeAdd(a: Float32Array, b: Float32Array, out: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = a[i % a.length] + b[i % b.length];  // modulo for suffix broadcasting
  }
}

export const addKernel: CPUKernel = (_node, inputs, outputs) => {
  executeAdd(
    inputs[0].buffer as Float32Array,
    inputs[1].buffer as Float32Array,
    outputs[0].buffer as Float32Array,
  );
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

Add test cases in `tests/ops/<opname>.test.ts` using the `BACKENDS` array from `tests/helpers.ts`.
Run against CPU first (oracle). Exclude backends that don't yet have the kernel via filtering.

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

**Register in `packages/backend-wasm/src/kernels/registry.ts`:**

```ts
import { sigmoidKernel } from './elementwise/sigmoid';
// ...
['Sigmoid', sigmoidKernel],
```

**Rebuild WASM** after changing Rust:

```sh
make wasm
# or: cd packages/backend-wasm && wasm-pack build rust --target bundler --out-dir ../pkg
```

---

## Step 6: WebGPU kernel

**WGSL shader** — `packages/backend-webgpu/src/kernels/<category>/<OpName>.wgsl`:

```wgsl
@group(0) @binding(0) var<storage, read>       A   : array<f32>;
@group(0) @binding(1) var<storage, read_write> Out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&Out)) { return; }
  Out[i] = 1.0 / (1.0 + exp(-A[i]));
}
```

**TypeScript kernel** — `packages/backend-webgpu/src/kernels/<category>/<OpName>.ts`:

Implements the `WebGPUKernel` interface from `../utils`. Three methods:

| Method | Responsibility |
| --- | --- |
| `createPipeline(device)` | Compile the WGSL shader into a `GPUComputePipeline`. |
| `buildBindGroupEntries(device, node, inputs, outputs)` | Wire buffers (and any uniform buffers for shape metadata) to binding slots. |
| `getDispatch(node, inputs, outputs)` | Return `[x, y, z]` workgroup counts. |

```ts
import source from './sigmoid.wgsl?raw';
import { WebGPUKernel, elementwiseBindGroupEntries, flatDispatch } from '../utils';

export const sigmoidKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: source, label: 'SigmoidShader' }), entryPoint: 'main' },
      label: 'SigmoidPipeline',
    });
  },
  buildBindGroupEntries(_device, _node, inputs, outputs) {
    return elementwiseBindGroupEntries(inputs, outputs);
  },
  getDispatch(_node, _inputs, outputs) {
    return flatDispatch(outputs);
  },
};
```

**Ops that need shape metadata** (e.g. MatMul needs M/K/N): use `createUniformBuffer` from `../utils` inside `buildBindGroupEntries` and push the uniform to the entries. See `matmulKernel` for the pattern.

**Register in `packages/backend-webgpu/src/kernels/registry.ts`:**

```ts
import { sigmoidKernel } from './elementwise/sigmoid';
// ...
['Sigmoid', sigmoidKernel],
```

No changes to `backend.ts` are needed — `execute()` is fully generic.

---

## Step 7: Parity test

Once all target backends support the op, add a `describe` block in `tests/backend/consistency.test.ts` that runs the same graph on each backend and asserts outputs match CPU within `1e-5`.
