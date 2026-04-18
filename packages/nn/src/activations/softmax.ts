import { Tensor, sum, mul, sub } from '@webtensor/core';

/**
 * Numerically-stable softmax along `axis` (default: last). IR op: `Softmax`.
 */
export function softmax(a: Tensor, axis = -1): Tensor {
  const rank = a.shape.length;
  const normalizedAxis = axis < 0 ? rank + axis : axis;

  return new Tensor({
    shape: a.shape as number[],
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Softmax',
      inputs: [a],
      attributes: { axis: normalizedAxis },
      backward: (grad) => {
        // d/da softmax = s * (grad - sum(grad * s, axis, keepdim=true))
        const s = softmax(a, axis);
        const dot = sum(mul(grad, s), normalizedAxis, /* keepdim */ true);
        return [mul(s, sub(grad, dot))];
      },
    },
  });
}
