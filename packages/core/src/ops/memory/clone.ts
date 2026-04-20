import { Tensor } from '../../tensor';

/**
 * Deep copy of the tensor's data into fresh storage.
 * @category Memory
 */
export function clone(a: Tensor): Tensor {
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
