import { Tensor, mul, sub, tensor } from '@webtensor/core';

/** Element-wise logistic `1 / (1 + exp(-a))`. */
export function sigmoid(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Sigmoid',
      inputs: [a],
      backward: (grad) => {
        const s = sigmoid(a);
        return [mul(grad, mul(s, sub(tensor([1]), s)))];
      },
    },
  });
}
