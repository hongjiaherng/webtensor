import { Tensor } from '../../tensor';

/** Materialize a strided/view tensor into a fresh contiguous allocation. */
export function contiguous(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Contiguous',
      inputs: [a],
      backward: (grad) => [grad],
    },
  });
}
