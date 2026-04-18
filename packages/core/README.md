# @webtensor/core

User-facing tensor API for [webtensor](https://github.com/hongjiaherng/webtensor) — `Tensor`, op functions, autograd, and `compileGraph()`.

## Install

```sh
npm install @webtensor/core @webtensor/runtime @webtensor/backend-cpu
```

`@webtensor/ir` and `@webtensor/runtime` are re-exported from `@webtensor/core`.

## Usage

```ts
import { add, matmul, tensor, compileGraph } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';

const a = tensor([[1, 2], [3, 4]]);
const b = tensor([[5, 6], [7, 8]]);
const c = matmul(add(a, b), a);

const engine = new Engine(await CPUBackend.create());
await engine.evaluate(compileGraph([c]));
const out = (await engine.get(c.id)) as Float32Array;
```

Ops are also chainable: `a.add(b).matmul(a)`.

## Docs

- [System walkthrough](https://hongjiaherng.github.io/webtensor/docs/onboarding/system-walkthrough)
- [Package deep-dive](https://hongjiaherng.github.io/webtensor/docs/onboarding/package-deep-dive)
- [Roadmap](https://hongjiaherng.github.io/webtensor/docs/roadmap)
