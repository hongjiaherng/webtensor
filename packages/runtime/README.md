# @webtensor/runtime

Execution engine for webtensor — graph traversal, tensor lifecycle, view ops, and the `Backend` interface that CPU/WASM/WebGPU backends implement.

Part of [webtensor](https://github.com/hongjiaherng/webtensor) — a tensor library that runs entirely in the browser with WebGPU/WASM/CPU backends.

## Install

```sh
npm install @webtensor/runtime @webtensor/ir
```

You'll typically also install `@webtensor/core` (for the graph-building API) and at least one backend.

## Usage

Build a graph with `@webtensor/core`, then evaluate it with an `Engine` + a `Backend`:

```ts
import { Engine } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';
import { add, tensor, compileGraph } from '@webtensor/core';

const engine = new Engine(new CPUBackend());

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

Or use the registry-based factory (after the backend's `registerBackend()` side-effect runs on import):

```ts
import { Engine } from '@webtensor/runtime';
import '@webtensor/backend-webgpu'; // registers 'webgpu'

const engine = await Engine.create('webgpu');
```

## API

**Engine**

- `new Engine(backend: Backend)`
- `static async create(device: string): Promise<Engine>` — uses registry
- `set(name, data, shape, dtype?)` — bind external input
- `async get(name): Promise<ArrayBufferView | undefined>` — read tensor
- `dispose(name)` — release tensor
- `async evaluate(graph: Graph)` — execute the compiled graph

**Backend interface** (implement this to add a custom backend)

- `allocate(shape, dtype): RuntimeTensor`
- `read(tensor): Promise<ArrayBufferView>`
- `write(tensor, data): void`
- `execute(node, inputs, outputs): void | Promise<void>`
- `dispose(tensor): void`

**Backend registry**

- `registerBackend(device: string, factory: () => Promise<Backend>)`

**Other exports**

- `RuntimeTensor`, `RuntimeStorage` — execution-level tensor model (PyTorch Tensor/Storage split)
- Stride utils: `getShapeSize`, `stridedIdx`, `broadcastStridesOf`, `isContiguous`, `computeContiguousStrides`
- Dtype utils: `bytesPerElement`, `typedArrayCtor`, `copyBuffer`, `TypedArray`
- View registry: `viewRegistry` (`Transpose`, `Slice`, `Unsqueeze`, `Squeeze`, `Permute`, `Expand` — all zero-copy)

## Docs

- [Package deep-dive — `packages/runtime`](https://hongjiaherng.github.io/webtensor/docs/onboarding/package-deep-dive)
- [Architecture](https://hongjiaherng.github.io/webtensor/docs/advanced/architecture)
