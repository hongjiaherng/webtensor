# Next Steps

Working context doc. The "Immediate task" section gets rewritten when the active task changes.

---

## Current State

**Verified working:**
- CPU backend — all 6 ops (Add, Sub, Mul, Div, MatMul, Transpose), reference implementation
- Autograd — reverse-mode diff for MatMul; forward pass for Add, Mul, Relu, Transpose
- Engine — topological sort, reference-counted disposal of intermediates, retained set for inputs/outputs/initializers
- WASM backend — same 6 ops as CPU, tensor memory in WASM heap, pointer-based kernel calls

**Scaffolded but unverified:**
- WASM in browser — tests run under Node via Vitest; the WASM backend has not been validated in a real browser
- WebGPU correctness — 4 ops (Add, Mul, MatMul, Transpose) exist but have never been numerically compared to CPU output

**Absent:**
- Relu kernel on any backend (defined in `core/ops.ts` but no kernel implementations)
- All other activation ops (Sigmoid, Tanh, Softmax)
- Broadcasting for binary ops — same-shape only; `[N,D] + [D]` not supported
- Batched MatMul (rank ≥ 3)
- Cross-backend parity tests
- Loss functions, optimizers, training loop
- devtools, ONNX parser

---

## Immediate Task: Cross-Backend Vertical Slice

**Goal:** `y = Relu(MatMul(x, W) + b)` produces matching outputs on CPU, WASM, and WebGPU.

**Acceptance criteria:**
1. A single test compiles the graph once and runs it on all three backends
2. WASM output matches CPU within `1e-5`
3. WebGPU output matches CPU within `1e-4` (GPU float tolerance)
4. Test is deterministic (fixed `x`, `W`, `b` values)

**What this requires:**
- Relu kernel on CPU, WASM (Rust + `relu_raw`), and WebGPU (WGSL)
- Broadcasting for Add: `[N, D] + [D]` (bias addition pattern)
- A parity test helper that runs the same graph on multiple backends and diffs the outputs

**Where to start:**
1. CPU Relu kernel (`packages/backend-cpu/src/kernels/elementwise/relu.ts`) + register + test
2. Write the parity helper (small utility in `tests/` that compares two `ArrayBufferView`s within tolerance)
3. WASM Relu (Rust `relu_raw` + extend `MinitensorWasmModule` + register) + rebuild WASM
4. WebGPU Relu (WGSL shader + pipeline factory + register) — no dispatch special case needed
5. Broadcasting in Add — update CPU, WASM, WebGPU `executeAdd` / `add_raw` / `add.wgsl` to handle `[N,D] + [D]`
6. Write the full vertical slice parity test

See [docs/adding-an-op.md](./adding-an-op.md) for the step-by-step procedure.

---

## Sharp Edges

These will bite you if you don't know them:

- **WebGPU dispatch is not fully registry-driven.** `WebGPUBackend.execute()` (`packages/backend-webgpu/src/backend.ts:99–131`) has `if (node.op === 'MatMul')` and `if (node.op === 'Transpose')` blocks that inject uniform buffers outside the kernel registry. New ops that need shape metadata passed as uniforms require a new block there.

- **WASM is unvalidated in a browser.** All WASM tests run under Node via Vitest. The `wasm-pack --target bundler` output has not been loaded in a real browser. There may be initialization or memory issues that Node masks.

- **`compileGraph()` uses `requiresGrad` to classify inputs vs initializers.** In `packages/core/src/compiler.ts`, a `Constant` node with `requiresGrad: true` is classified as a graph input; without it, as an initializer. This works for the current use cases but will break when weight tensors need to be both `requiresGrad: true` (for training) and initializers (for inference reuse). Needs a redesign before the training loop is built.

- **`Engine.evaluate()` is synchronous, but WebGPU `read()` is async.** `packages/runtime/src/engine.ts` calls `backend.execute()` synchronously. The `get()` method returns `ArrayBufferView | Promise<ArrayBufferView>`, but `evaluate()` itself does not await. This means callers must manually `await engine.get(outputId)` after `evaluate()`, and there is no guarantee that GPU work has completed. This is a latent inconsistency that will need a proper async execution model.

---

## Priority Order

| Phase | Work | Blocked by |
| --- | --- | --- |
| 2 | Relu + broadcasting Add + cross-backend parity test | nothing |
| 3 | Sub/Div on WebGPU; remaining activations (Sigmoid, Tanh); batched MatMul; Reshape, Concat | phase 2 |
| 4 | Loss functions (MSE, CrossEntropy); SGD + Adam optimizers; training loop in `core` | phase 3 |
| 5 | `devtools` package: graph viz, weight/activation inspector, real-time loss curve | parallel with 4 |
| 6 | `onnx` package: protobuf parser, ONNX op → IR mapping | phase 4 |
| 7 | Kernel fusion, buffer pooling, async engine execution model | phase 6 |
