import { Tensor } from '@webtensor/core';

/** Element-wise `max(0, a)`. Backward via the dedicated `ReluGrad` op for efficiency. */
export function relu(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Relu',
      inputs: [a],
      backward: (grad) => [
        new Tensor({
          shape: a.shape,
          dtype: a.dtype,
          device: a.device,
          requiresGrad: false,
          ctx: { op: 'ReluGrad', inputs: [grad, a] },
        }),
      ],
    },
  });
}
