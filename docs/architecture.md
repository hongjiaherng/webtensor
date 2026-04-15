# Minitensor Architecture Notes

This project should stay small enough that one person can understand the full path from tensor authoring to backend execution.

## Recommended Package Boundaries

The current package split is a good foundation:

```text
packages/
  core/             User-facing Tensor API, eager ops, autograd graph capture
  ir/               Backend-neutral graph/value/node schema
  runtime/          Graph execution, tensor lifetime, backend dispatch contract
  backend-cpu/      Reference backend and correctness oracle
  backend-wasm/     Browser CPU acceleration backend
  backend-webgpu/   GPU backend
```

That separation is worth keeping. The main rule is:

```text
core can build graphs, but should not know how a backend stores memory.
ir can describe graphs, but should not know about autograd or devices.
runtime can execute graphs, but should not contain op math.
backends can run kernels, but should not define user-facing tensor semantics.
```

## Suggested Near-Term Structure

The current structure is already close. The next cleanups should be evolutionary:

```text
packages/
  core/
    src/
      tensor.ts
      tensor_init.ts
      ops/
        elementwise.ts
        linear.ts
        shape.ts
      autograd/
        engine.ts
      compiler/
        compileGraph.ts

  ir/
    src/
      types.ts
      shape-inference.ts
      validate.ts

  runtime/
    src/
      backend.ts
      engine.ts
      execution-plan.ts

  backend-cpu/
    src/
      backend.ts
      kernels/

  backend-wasm/
    src/
      backend.ts
      module.ts
      kernels/
    rust/
      src/

  backend-webgpu/
    src/
      backend.ts
      device.ts
      kernels/
      shaders/
```

Do not rush to create every folder. Add these only when the existing files become crowded.

## Backend Contract

The current `Backend` interface is the right idea:

```ts
interface Backend {
  allocate(...)
  read(...)
  write(...)
  execute(...)
  dispose(...)
}
```

For the next phase, consider making two concepts explicit:

```text
RuntimeTensor = backend-owned memory handle
Kernel = implementation of one IR op for one backend
```

That helps keep `Engine` simple. It should plan execution and call kernels, not understand MatMul dimensions beyond what the IR and backend contract require.

## Kernel Registry

Backends should maintain an explicit op registry instead of growing `switch` statements inside `execute()`.

```text
node.op -> backend-specific kernel runner
```

That gives each backend one obvious place to answer:

```text
Which ops do I support?
How does this op map to my implementation?
What backend-specific metadata does this op need?
```

The current direction is:

```text
backend-cpu/src/kernels/registry.ts
backend-wasm/src/kernels/registry.ts
backend-webgpu/src/kernels/registry.ts
```

## WASM Backend Assessment

The WASM backend now uses a more conventional runtime shape:

```text
WASM owns tensor memory
RuntimeTensor stores a pointer handle
write() copies JS data into WASM memory
execute() calls Rust kernels with pointers
read() copies data back into a JS Float32Array
dispose() frees the WASM allocation
```

This is a better foundation for adding many ops because tensors no longer cross the JS/WASM boundary on every kernel call.

The current package layout is acceptable:

```text
backend-wasm/
  src/   TypeScript package users import
  rust/  Rust crate compiled by wasm-pack
  pkg/   generated wasm-bindgen output
```

For a package whose public API is TypeScript, this is a normal shape. If the Rust crate grows large enough to be developed independently, then split it into a workspace-level `crates/minitensor-wasm/` later. Do not split it just for convention.

## WASM Generated Package Boundary

The generated `pkg/` folder should be treated as build output with a small wrapper boundary:

```text
backend-wasm/src/module.ts
```

That file is the only TypeScript source that should know the generated import path:

```ts
import initWasm from '../pkg/minitensor_wasm';
```

Kernel code should depend on `module.ts`, not import from `pkg/` directly. This keeps generated code contained.

## Recommendation For WASM

Keep Rust and TypeScript in the same `backend-wasm` package for now. The next WASM improvements should be allocator discipline, dtype support, and tests that prove disposed tensors do not remain readable.

## Testing Strategy

Keep CPU as the reference backend. Every op should have:

```text
same graph -> CPU result
same graph -> WASM result
same graph -> WebGPU result, when WebGPU is available
```

Browser/WebGPU tests should stay in Vitest browser mode. Bun's native test runner should not be the main test command, because it will scan `_archive` and ignore the Vitest include boundary.

## What To Build Next

The best next feature is not ONNX yet. Build a tiny visual graph demo:

```text
y = Relu(MatMul(x, W) + b)
```

It should show:

```text
graph nodes
tensor shapes
backend selected
output values
optional timing per node
```

That gives the project its identity: a learning-friendly tensor library where execution and visualization grow together.

## Longer-Term API Direction

The authoring layer can mimic PyTorch without copying its internals:

```ts
const x = tensor([[1, 2]]);
const w = tensor([[3], [4]], { requiresGrad: true });
const y = x.matmul(w).relu();
```

The runtime layer should remain ONNX-friendly:

```text
ONNX model -> internal IR -> execution plan -> backend kernels
```

That means PyTorch-like ergonomics belong in `core`, while ONNX import belongs in a future `onnx` package that produces the same IR used by hand-authored tensors.
