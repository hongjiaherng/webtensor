# @webtensor/optim

Host-side optimizers for [webtensor](https://github.com/hongjiaherng/webtensor). Parameters live as `Tensor` values in the JS runtime; `optimizer.step(params, grads)` mutates them in place between `compile()`d training steps.

## Install

```sh
npm install @webtensor/optim @webtensor/core @webtensor/runtime @webtensor/backend-cpu
```

## Usage

```ts
import { randn } from '@webtensor/core';
import { SGD } from '@webtensor/optim';

const W = randn([2, 8], { requiresGrad: true, std: 0.5 });
const b = randn([8], { requiresGrad: true });

const opt = new SGD(0.1);
for (let i = 0; i < 1000; i++) {
  const { loss, dW, db } = await step({ x: xBatch, y: yBatch });
  opt.step([W, b], [dW, db]);
}
```

Available: `SGD`. Adam is planned — see the [roadmap](https://hongjiaherng.github.io/webtensor/docs/roadmap).

## Docs

- [Package deep-dive](https://hongjiaherng.github.io/webtensor/docs/onboarding/package-deep-dive)
