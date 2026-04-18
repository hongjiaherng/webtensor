# @webtensor/backend-cpu

Pure TypeScript kernels for webtensor — no WebGPU, no WASM, no native deps. Synchronous `execute()`. Use this as the correctness oracle and for environments without GPU/WASM support.

Part of [webtensor](https://github.com/hongjiaherng/webtensor) — a tensor library that runs entirely in the browser with WebGPU/WASM/CPU backends.

## Install

```sh
npm install @webtensor/backend-cpu @webtensor/runtime @webtensor/core
```

## Usage

```ts
import { CPUBackend } from '@webtensor/backend-cpu';
import { Engine } from '@webtensor/runtime';
import { add, tensor, compileGraph } from '@webtensor/core';

const engine = new Engine(await CPUBackend.create());

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

Importing this package also auto-registers the `'cpu'` backend factory, so you can equivalently use `await Engine.create('cpu')`.

## API

- `CPUBackend` — implements `Backend` from `@webtensor/runtime`
  - `await CPUBackend.create()` (matches the async factory convention shared by all backends; CPU has no real async setup)
  - `execute()` is **synchronous** — kernels run immediately on `TypedArray` storage

## Supported ops

Add, Sub, Mul, Div, MatMul (2D), Relu, ReluGrad, Neg, Exp, Log, Sqrt, Abs, Pow, Sigmoid, Tanh, Contiguous — 16 kernels with stride-aware addressing.

## Implementation notes

Plain TypeScript scalar loops over `Float32Array` (and `Int32Array` / `Uint8Array` for non-float32 dtype allocation). Strided indexing via `stridedIdx(flat, rank, shape, strides, offset)` from `@webtensor/runtime`. No worker threads, no SIMD intrinsics — correctness-first; this is the parity oracle.

## Docs

- [Backends and kernels](https://hongjiaherng.github.io/webtensor/docs/onboarding/backends-and-kernels)
- [Adding an op](https://hongjiaherng.github.io/webtensor/docs/advanced/adding-an-op)
