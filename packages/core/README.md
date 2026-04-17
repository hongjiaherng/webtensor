# @webtensor/core

User-facing tensor API for webtensor — `Tensor` class, op functions, autograd, and `compileGraph()`. This is the package most users start with.

Part of [webtensor](https://github.com/hongjiaherng/webtensor) — a tensor library that runs entirely in the browser with WebGPU/WASM/CPU backends.

## Install

```sh
# Core + a backend (pick at least one)
npm install @webtensor/core @webtensor/runtime @webtensor/backend-cpu
```

`@webtensor/ir` and `@webtensor/runtime` are required peer-style deps and are re-exported from `@webtensor/core` for convenience.

## Usage

```ts
import { add, matmul, tensor, compileGraph } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';

const a = tensor([
  [1, 2],
  [3, 4],
]);
const b = tensor([
  [5, 6],
  [7, 8],
]);
const c = matmul(add(a, b), a); // builds a graph; nothing runs yet

const engine = new Engine(new CPUBackend());
await engine.evaluate(compileGraph([c]));
const out = (await engine.get(c.id)) as Float32Array;
```

Ops are also chainable as methods: `a.add(b).matmul(a)`.

## API

**Tensor factories**

- `tensor(data, options?)` — from nested array literal
- `zeros(shape, options?)`
- `ones(shape, options?)`
- `InitOptions = { shape?, dtype?, device?, requiresGrad? }`

**Tensor class** (24+ chainable methods including all ops below)

- Properties: `id`, `shape`, `strides`, `size`, `dtype`, `device`, `requiresGrad`, `grad?`
- `backward()` — accumulates gradients
- Inspection: `dim()`, `numel()`, `isContiguous()`, `stride()`

**Ops** (also exported as standalone functions)

| Category         | Ops                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| Binary           | `add`, `sub`, `mul`, `div`                                                                      |
| Linalg           | `matmul` (2D only)                                                                              |
| Unary math       | `neg`, `exp`, `log`, `sqrt`, `abs`, `pow`                                                       |
| Activations      | `relu`, `sigmoid`, `tanh`                                                                       |
| View (zero-copy) | `transpose`, `reshape`, `view`, `slice`, `unsqueeze`, `squeeze`, `permute`, `expand`, `flatten` |
| Memory           | `contiguous`, `clone`, `detach`                                                                 |

**Compiler**

- `compileGraph(outputs: Tensor[]): Graph` — trace the graph, classify constants vs inputs, compute consumer references for ref-count GC

**Re-exports** — All of `@webtensor/ir` and `@webtensor/runtime` are re-exported.

## Docs

- [System walkthrough](https://hongjiaherng.github.io/webtensor/docs/onboarding/system-walkthrough) — end-to-end example
- [Package deep-dive — `packages/core`](https://hongjiaherng.github.io/webtensor/docs/onboarding/package-deep-dive)
- [Roadmap](https://hongjiaherng.github.io/webtensor/docs/roadmap) — what's missing (softmax, reduce, optimizers, training loop)
