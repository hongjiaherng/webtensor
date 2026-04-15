# Adding a New Op

Checklist for adding a kernel across all three backends. Follow in order — CPU establishes the correctness oracle before WASM or WebGPU are added.

---

## Step 1: Update the kernel support matrix

In `README.md`, add a row to the **Kernel Support Matrix** table with `—` for all backends. Update cells to `yes` as each backend is completed.

---

## Step 2: CPU kernel

**File location:** `packages/backend-cpu/src/kernels/<category>/<OpName>.ts`

Categories: `elementwise/`, `linear/`, `shape/`, `reduction/` (create new category if none fit).

**Unary elementwise pattern:**

```ts
export function executeRelu(a: Float32Array, out: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.max(0, a[i]);
  }
}
```

**Binary elementwise pattern** (see `add.ts`, `mul.ts`):

```ts
export function executeAdd(a: Float32Array, b: Float32Array, out: Float32Array): void {
  const aScalar = a.length === 1;
  const bScalar = b.length === 1;
  for (let i = 0; i < out.length; i++) {
    out[i] = (aScalar ? a[0] : a[i]) + (bScalar ? b[0] : b[i]);
  }
}
```

**Register in `packages/backend-cpu/src/kernels/registry.ts`:**

```ts
// Unary:
['Relu', (_node, inputs, outputs) => {
  executeRelu(inputs[0].buffer as Float32Array, outputs[0].buffer as Float32Array);
}],

// Binary elementwise shorthand:
['Add', binaryElementwise(executeAdd)],
```

---

## Step 3: CPU test

Add a test in `tests/` (or extend an existing suite) that runs the op on the CPU backend and asserts exact output values. This is the oracle all other backends are compared against.

Pattern: create a small graph with `compileGraph()`, run it via `Engine` with `CPUBackend`, read the output, assert values.

---

## Step 4: WASM kernel

**Rust function** in `packages/backend-wasm/rust/src/lib.rs` (or a submodule):

```rust
// Slice-based (used for testing in Rust):
#[wasm_bindgen]
pub fn relu(inp: &[f32], out: &mut [f32]) {
    for (o, &a) in out.iter_mut().zip(inp.iter()) {
        *o = a.max(0.0);
    }
}

// Pointer-based (used at runtime — avoids copy per call):
#[wasm_bindgen]
pub fn relu_raw(inp_ptr: *const f32, out_ptr: *mut f32, len: usize) {
    let inp = unsafe { std::slice::from_raw_parts(inp_ptr, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
    relu(inp, out);
}
```

**Extend `MinitensorWasmModule`** in `packages/backend-wasm/src/module.ts`:

```ts
export interface MinitensorWasmModule extends InitOutput {
  // ... existing fields ...
  readonly relu_raw: (inpPtr: number, outPtr: number, len: number) => void;
}
```

**Register in `packages/backend-wasm/src/kernels/registry.ts`:**

```ts
['Relu', (module, _node, inputs, outputs) => {
  const inp = handleOf(inputs[0]);
  const out = handleOf(outputs[0]);
  module.relu_raw(inp.ptr, out.ptr, out.elements);
}],
```

**Rebuild WASM** after changing Rust:

```sh
cd packages/backend-wasm && wasm-pack build --target bundler
```

---

## Step 5: WebGPU kernel

**WGSL shader** — create `packages/backend-webgpu/src/kernels/<category>/<OpName>.wgsl`:

```wgsl
@group(0) @binding(0) var<storage, read>       inp : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&out)) { return; }
  out[i] = max(0.0, inp[i]);
}
```

**TypeScript wrapper** — create `packages/backend-webgpu/src/kernels/<category>/<OpName>.ts`:

```ts
import shaderSource from './<OpName>.wgsl?raw';

export function getReluPipeline(device: GPUDevice): GPUComputePipeline {
  return device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: shaderSource }),
      entryPoint: 'main',
    },
  });
}
```

**Register in `packages/backend-webgpu/src/kernels/registry.ts`:**

```ts
['Relu', getReluPipeline],
```

---

## Step 6: WebGPU dispatch — does it need a special case?

`WebGPUBackend.execute()` in `packages/backend-webgpu/src/backend.ts` auto-handles two cases:

- **Elementwise (default path):** inputs at bindings `0..n-1`, output at binding `n`, 1D dispatch at `ceil(elements / 64)` workgroups. No special case needed for unary or binary ops without shape metadata.
- **Ops needing a uniform buffer** (shape dimensions passed to the shader): add an `else if (node.op === 'YourOp')` block to inject the uniform at binding `inputs.length + 1` and set the correct workgroup geometry. See the existing `MatMul` and `Transpose` blocks as templates.

Relu is elementwise — no special case needed. Softmax (needs an axis), batched MatMul, and Reshape (needs target shape) will need uniform blocks.

---

## Step 7: Parity test

Once all three backends support the op, add a cross-backend parity test. The pattern: run the same graph on CPU, WASM, and WebGPU, then assert that WASM and WebGPU outputs match CPU within tolerance (e.g. `1e-5` for float32).
