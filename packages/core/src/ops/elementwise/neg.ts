import { Tensor } from '../../tensor';

/**
 * Element-wise negation: `-a`.
 * @category Elementwise
 */
export function neg(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Neg',
      inputs: [a],
      backward: (grad) => [neg(grad)],
    },
  });
}
