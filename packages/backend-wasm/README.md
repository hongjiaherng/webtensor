# @webtensor/backend-wasm

Rust kernels compiled to WebAssembly for webtensor. The `.wasm` binary is **base64-inlined into the JS bundle at build time**, so consumers don't need `vite-plugin-wasm` or any other bundler config — just install and import.

Part of [webtensor](https://github.com/hongjiaherng/webtensor) — a tensor library that runs entirely in the browser with WebGPU/WASM/CPU backends.

## Install

```sh
npm install @webtensor/backend-wasm @webtensor/runtime @webtensor/core
```

No bundler plugins required. Works with Vite, webpack, esbuild, bun, parcel, etc.

## Usage

```ts
import { WASMBackend } from '@webtensor/backend-wasm';
import { Engine } from '@webtensor/runtime';
import { add, tensor, compileGraph } from '@webtensor/core';

const backend = await WASMBackend.create(); // async — instantiates WASM
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

Importing this package also auto-registers the `'wasm'` backend factory, so you can equivalently use `await Engine.create('wasm')`.

## API

- `WASMBackend` — implements `Backend` from `@webtensor/runtime`
  - `static async create(): Promise<WASMBackend>` — instantiates the WASM module (cached; only happens once per process)

## Supported ops

Add, Sub, Mul, Div, MatMul (2D), Relu, ReluGrad, Neg, Exp, Log, Sqrt, Abs, Pow, Sigmoid, Tanh, Contiguous — 16 kernels.

## Implementation notes

- **Memory model**: `Vec` leak/reclaim across the JS/WASM boundary. `alloc_f32(len)` does `Vec::with_capacity(len)` + `mem::forget`; `free_f32(ptr, len)` reconstructs via `Vec::from_raw_parts` to drop. No memory pool.
- **Kernel style**: Naive scalar `for i in 0..total { strided_idx; apply; }` loops. No SIMD, no `rayon` — sole crate dependency is `wasm-bindgen 0.2.89`.
- **Special kernel**: `relu_grad_raw` takes contiguous-only inputs (no meta buffer) since the autograd engine pre-allocates fresh contiguous tensors.
- **Matmul**: Naive 3-nested O(MNK), 2D only. No tiling.
- **Bundle**: The WASM binary is base64-encoded into `dist/index.js` by a Bun build plugin in `build.ts`. The published package contains only `dist/`; the wasm-pack output (`pkg/`) is build-time only.

## Docs

- [Backends and kernels](https://hongjiaherng.github.io/webtensor/docs/onboarding/backends-and-kernels)
- [Adding an op](https://hongjiaherng.github.io/webtensor/docs/advanced/adding-an-op)
