import { Tensor } from '../../tensor';

/**
 * Element-wise `max(0, a)`. Backward via the dedicated `ReluBackward` op for efficiency.
 * @category Activation
 */
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
          ctx: { op: 'ReluBackward', inputs: [grad, a] },
        }),
      ],
    },
  });
}
