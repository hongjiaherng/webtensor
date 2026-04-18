# @webtensor/ir

Pure data types for the [webtensor](https://github.com/hongjiaherng/webtensor) computation graph — `Node`, `Value`, `Graph`, `DType`. No devices, no gradients, no execution logic.

## Install

Usually a transitive dep of `@webtensor/core` and the backends. Install directly only if you're authoring a custom backend or graph rewriter.

```sh
npm install @webtensor/ir
```

## Usage

```ts
import type { Graph, Node, Value, DType } from '@webtensor/ir';
import { computeContiguousStrides } from '@webtensor/ir';

const strides = computeContiguousStrides([2, 3, 4]); // [12, 4, 1]
```

## Docs

- [Package deep-dive](https://hongjiaherng.github.io/webtensor/docs/onboarding/package-deep-dive)
- [IR reference](https://hongjiaherng.github.io/webtensor/docs/advanced/ir-reference)
