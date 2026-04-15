# Architecture: Design Decisions

Non-obvious design decisions that aren't visible from reading the code. For the full picture (diagrams, package table, roadmap) see [README.md](../README.md).

---

## Package Boundaries

The boundary rule exists to keep each package independently testable and replaceable:

- `core` can build graphs but must not know how a backend stores memory — otherwise swapping backends would require changing user-facing code.
- `ir` can describe computation but must not know about autograd or devices — the same graph must be producible from hand-authored tensors and from an ONNX parser without either knowing about the other.
- `runtime` can execute graphs but must not contain op math — otherwise adding a backend would mean touching the engine.
- `backends` can run kernels and own memory but must not define user-facing tensor semantics — user code imports from `core`, not from backend packages.

## Backend Contract

Two concepts must remain distinct:

```text
RuntimeTensor  = backend-owned memory handle (shape + dtype + opaque buffer)
Kernel         = implementation of one IR op for one backend
```

`Engine` should only plan execution order and call kernels. It must not understand MatMul dimensions, WGSL workgroup geometry, or WASM pointer arithmetic. When `Engine.execute()` receives a node, it allocates outputs from the backend and delegates everything else. If `Engine` is growing op-specific logic, that logic belongs in the backend.

## Kernel Registry Pattern

Each backend owns a `Map<string, KernelFn>` keyed by `node.op`. This replaces growing `switch` statements in `execute()` and gives each backend one canonical place to answer:

- Which ops do I support?
- How does this op map to my implementation?
- What backend-specific state does this op need?

Registry files:

- `packages/backend-cpu/src/kernels/registry.ts`
- `packages/backend-wasm/src/kernels/registry.ts`
- `packages/backend-webgpu/src/kernels/registry.ts`

**Exception:** `WebGPUBackend.execute()` in `packages/backend-webgpu/src/backend.ts` currently has op-specific dispatch logic baked in for `MatMul` and `Transpose`. These ops need a uniform buffer (shape metadata injected at binding `inputs.length + 1`) and non-standard workgroup geometry. Until that dispatch is refactored into the kernel registry, any new WebGPU op that requires a uniform buffer needs a new `if (node.op === '...')` block there. See [docs/adding-an-op.md](./adding-an-op.md) for the full procedure.

## WASM Backend: Memory Model

Tensor memory lives in the WASM heap, not in JS. The lifecycle:

```text
allocate()  →  module.alloc_f32(size)         returns a pointer; wraps in WasmTensorHandle
write()     →  copies JS TypedArray into WASM heap via module.memory.buffer view
execute()   →  calls Rust kernel with raw pointers (no JS/WASM data crossing per element)
read()      →  copies WASM heap slice back into a new JS Float32Array
dispose()   →  module.free_f32(ptr, elements)  frees the WASM allocation
```

Tensors no longer cross the JS/WASM boundary on every kernel call. This is the key difference from a naive implementation where each op takes `&[f32]` slices — that pattern copies data on every call. The `_raw` suffix on Rust functions (e.g., `add_raw`) marks the pointer-based variants used at runtime.

## WASM Generated Package Boundary

`packages/backend-wasm/pkg/` is build output from `wasm-pack`. It must not be imported directly by kernel code.

`packages/backend-wasm/src/module.ts` is the only TypeScript file that imports from `pkg/`:

```ts
import initWasm from '../pkg/minitensor_wasm';
```

All kernel code imports from `module.ts`. This contains the `MinitensorWasmModule` interface, the `WasmTensorHandle` type, `loadWasmModule()`, and `getF32View()`. If `wasm-pack` regenerates `pkg/` with a different structure, only `module.ts` needs updating.

## Testing Strategy

CPU is the correctness oracle. Every op must pass:

```text
same graph  →  CPU result
same graph  →  WASM result     (must match CPU within tolerance)
same graph  →  WebGPU result   (when WebGPU is available)
```

Tests run in Vitest browser mode (Playwright, Chromium with `--enable-unsafe-webgpu`). Do not use `bun test` as the primary test command — it will ignore Vitest's `include` boundary and may pick up unintended files.

WASM tests currently run under Node via Vitest. The WASM backend has not been validated in a real browser. Any test claiming WASM correctness is only valid under Node until browser validation is added.
