import { Tensor } from '../../tensor';

/**
 * Element-wise `|a|`.
 *
 * NOTE: backward is not implemented (would require a `sign(a)` op).
 */
export function abs(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Abs',
      inputs: [a],
    },
  });
}
