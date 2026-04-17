# @webtensor/ir

Pure data types for webtensor's computation graph IR — `Node`, `Value`, `Graph`, `DType`. No devices, no gradients, no execution logic.

Part of [webtensor](https://github.com/hongjiaherng/webtensor) — a tensor library that runs entirely in the browser with WebGPU/WASM/CPU backends.

## Install

You usually don't install this directly — it's a transitive dep of [`@webtensor/core`](https://www.npmjs.com/package/@webtensor/core) and the backends. Install only if you're authoring a custom backend or graph rewriter.

```sh
npm install @webtensor/ir
```

## Usage

```ts
import type { Graph, Node, Value, DType } from '@webtensor/ir';
import { computeContiguousStrides } from '@webtensor/ir';

const strides = computeContiguousStrides([2, 3, 4]); // [12, 4, 1]
```

## API

| Export                            | Kind     | Description                                                                         |
| --------------------------------- | -------- | ----------------------------------------------------------------------------------- |
| `Node`                            | type     | An op in the graph: `{ id, op, inputs, outputs, attributes?, name? }`               |
| `Value`                           | type     | Tensor metadata: `{ name, shape, dtype, data?, producer?, consumers?, debugName? }` |
| `Graph`                           | type     | `{ nodes, values, inputs, outputs, initializers, name?, opset? }`                   |
| `DType`                           | type     | `'float32' \| 'int32' \| 'bool'`                                                    |
| `AttributeValue`                  | type     | Union of scalar/array/buffer attribute values                                       |
| `computeContiguousStrides(shape)` | function | C-order (row-major) strides; innermost = 1                                          |

## Docs

- [Package deep-dive — `packages/ir`](https://hongjiaherng.github.io/webtensor/docs/onboarding/package-deep-dive)
- [IR reference](https://hongjiaherng.github.io/webtensor/docs/advanced/ir-reference)
