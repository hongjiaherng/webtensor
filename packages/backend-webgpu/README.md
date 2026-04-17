# @webtensor/backend-webgpu

WGSL compute shaders for webtensor — runs on the browser's GPU via the WebGPU API.

Part of [webtensor](https://github.com/hongjiaherng/webtensor) — a tensor library that runs entirely in the browser with WebGPU/WASM/CPU backends.

## Install

```sh
npm install @webtensor/backend-webgpu @webtensor/runtime @webtensor/core
```

Requires a browser (or environment) with WebGPU support — Chromium 113+, Edge 113+, Safari 18+, Firefox 141+. Detect with `'gpu' in navigator`.

## Usage

```ts
import { WebGPUBackend } from '@webtensor/backend-webgpu';
import { Engine } from '@webtensor/runtime';
import { add, tensor, compileGraph } from '@webtensor/core';

if (!('gpu' in navigator)) {
  throw new Error('WebGPU not supported in this environment');
}

const backend = await WebGPUBackend.create(); // requests adapter + device
const engine = new Engine(backend);

const a = tensor([
  [1, 2],
  [3, 4],
]);
const b = tensor([
  [5, 6],
  [7, 8],
]);
const c = add(a, b);

await engine.evaluate(compileGraph([c]));
const out = (await engine.get(c.id)) as Float32Array; // [6, 8, 10, 12]
```

Importing this package also auto-registers the `'webgpu'` backend factory, so you can equivalently use `await Engine.create('webgpu')`.

## API

- `WebGPUBackend` — implements `Backend` from `@webtensor/runtime`
  - `static async create(): Promise<WebGPUBackend>` — calls `navigator.gpu.requestAdapter()` then `adapter.requestDevice()`. Throws if WebGPU is unavailable or no adapter is found.

## Supported ops

Add, Sub, Mul, Div, MatMul (2D), Relu, ReluGrad, Neg, Exp, Log, Sqrt, Abs, Pow, Sigmoid, Tanh, Contiguous — 16 kernels.

## Implementation notes

- **dtype support**: `float32` only currently. The backend allocates `Float32Array`-typed GPU storage for all dtypes; `int32` / `bool` data round-trips but no kernels run on them yet.
- **Pipeline cache**: Compute pipelines are compiled lazily on first use and cached per op.
- **TensorMeta uniform**: 80 bytes, 20 × u32 — `{ rank, offset, _pad×2, shape: array<vec4<u32>, 2>, strides: array<vec4<u32>, 2> }`. Supports up to rank-8 tensors. `packMeta(tensor, outShape?)` produces broadcast-aware strides (size-1 dims → stride 0).
- **Workgroup sizes**: `@workgroup_size(64)` for elementwise; `@workgroup_size(8, 8)` for matmul.
- **Kernel style**: One thread per output element. Matmul accumulates each output cell via a scalar K-dim loop — no shared-memory tiling. Naive but easy to verify against the CPU oracle.
- **Cleanup**: Temp meta buffers are tracked per-execute and destroyed after `device.queue.onSubmittedWorkDone()`.
- **WGSL gotcha**: `meta` is a reserved keyword — uniforms in this package are always named `u_meta` / `u_meta_a` / `u_meta_b`. Bare `meta` causes silent pipeline compile failure (the error pipeline produces all-zero output).

## Docs

- [Backends and kernels](https://hongjiaherng.github.io/webtensor/docs/onboarding/backends-and-kernels)
- [Adding an op](https://hongjiaherng.github.io/webtensor/docs/advanced/adding-an-op)
