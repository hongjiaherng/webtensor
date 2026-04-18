# @webtensor/nn

Neural-network building blocks for [webtensor](https://github.com/hongjiaherng/webtensor) — activations (`relu`, `sigmoid`, `tanh`, `softmax`) and losses (`mseLoss`). All are thin graph-level functions that compose with the core op set.

## Install

```sh
npm install @webtensor/nn @webtensor/core @webtensor/runtime @webtensor/backend-cpu
```

## Usage

```ts
import { tensor, matmul, compile, grad } from '@webtensor/core';
import { relu, mseLoss } from '@webtensor/nn';

const step = await compile(
  ({ x, y, W, b }) => {
    const pred = relu(matmul(x, W).add(b));
    const loss = mseLoss(pred, y);
    return { loss, dW: grad(loss, W), db: grad(loss, b) };
  },
  { x: [4, 2], y: [4, 8], W: [2, 8], b: [8] },
  { device: 'cpu' },
);
```

## Docs

- [Package deep-dive](https://hongjiaherng.github.io/webtensor/docs/onboarding/package-deep-dive)
- [Autograd](https://hongjiaherng.github.io/webtensor/docs/onboarding/autograd)
