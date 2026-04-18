import { Tensor, mul, sub, tensor } from '@webtensor/core';

/** Element-wise `tanh(a)`. */
export function tanh(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Tanh',
      inputs: [a],
      backward: (grad) => {
        const t = tanh(a);
        return [mul(grad, sub(tensor([1]), mul(t, t)))];
      },
    },
  });
}
