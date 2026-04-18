# @webtensor/backend-cpu

Pure TypeScript kernels for [webtensor](https://github.com/hongjiaherng/webtensor) — the correctness oracle. Synchronous `execute()`, no GPU or WASM dependencies.

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

const a = tensor([[1, 2], [3, 4]]);
const b = tensor([[5, 6], [7, 8]]);
const c = add(a, b);

await engine.evaluate(compileGraph([c]));
const out = (await engine.get(c.id)) as Float32Array;
```

Importing this package auto-registers the `'cpu'` backend, so `await Engine.create('cpu')` also works.

## Docs

- [Backends and kernels](https://hongjiaherng.github.io/webtensor/docs/onboarding/backends-and-kernels)
- [Adding an op](https://hongjiaherng.github.io/webtensor/docs/advanced/adding-an-op)
