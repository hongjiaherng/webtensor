# @webtensor/runtime

Execution engine for [webtensor](https://github.com/hongjiaherng/webtensor) — graph traversal, tensor lifecycle, view ops, and the `Backend` interface implemented by CPU/WASM/WebGPU backends.

## Install

```sh
npm install @webtensor/runtime @webtensor/ir
```

You'll typically also install `@webtensor/core` and at least one backend.

## Usage

```ts
import { Engine } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';
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

Backends self-register on import, so `await Engine.create('webgpu')` works after `import '@webtensor/backend-webgpu'`.

## Docs

- [Package deep-dive](https://hongjiaherng.github.io/webtensor/docs/onboarding/package-deep-dive)
- [Architecture](https://hongjiaherng.github.io/webtensor/docs/advanced/architecture)
