# @webtensor/backend-wasm

Rust kernels compiled to WebAssembly for [webtensor](https://github.com/hongjiaherng/webtensor). The `.wasm` binary is base64-inlined into the JS bundle — no bundler plugins required.

## Install

```sh
npm install @webtensor/backend-wasm @webtensor/runtime @webtensor/core
```

Works with Vite, webpack, esbuild, bun, parcel out of the box.

## Usage

```ts
import { WASMBackend } from '@webtensor/backend-wasm';
import { Engine } from '@webtensor/runtime';
import { add, tensor, compileGraph } from '@webtensor/core';

const engine = new Engine(await WASMBackend.create());

const a = tensor([[1, 2], [3, 4]]);
const b = tensor([[5, 6], [7, 8]]);
const c = add(a, b);

await engine.evaluate(compileGraph([c]));
const out = (await engine.get(c.id)) as Float32Array;
```

Importing this package auto-registers the `'wasm'` backend, so `await Engine.create('wasm')` also works.

## Docs

- [Backends and kernels](https://hongjiaherng.github.io/webtensor/docs/onboarding/backends-and-kernels)
- [Adding an op](https://hongjiaherng.github.io/webtensor/docs/advanced/adding-an-op)
