# @webtensor/backend-webgpu

WGSL compute shaders for [webtensor](https://github.com/hongjiaherng/webtensor) — runs on the browser's GPU via the WebGPU API.

## Install

```sh
npm install @webtensor/backend-webgpu @webtensor/runtime @webtensor/core
```

Requires a browser with WebGPU support (Chromium 113+, Edge 113+, Safari 18+, Firefox 141+). Detect with `'gpu' in navigator`.

## Usage

```ts
import { WebGPUBackend } from '@webtensor/backend-webgpu';
import { Engine } from '@webtensor/runtime';
import { add, tensor, compileGraph } from '@webtensor/core';

const engine = new Engine(await WebGPUBackend.create());

const a = tensor([[1, 2], [3, 4]]);
const b = tensor([[5, 6], [7, 8]]);
const c = add(a, b);

await engine.evaluate(compileGraph([c]));
const out = (await engine.get(c.id)) as Float32Array;
```

Importing this package auto-registers the `'webgpu'` backend, so `await Engine.create('webgpu')` also works.

## Docs

- [Backends and kernels](https://hongjiaherng.github.io/webtensor/docs/onboarding/backends-and-kernels)
- [Adding an op](https://hongjiaherng.github.io/webtensor/docs/advanced/adding-an-op)
