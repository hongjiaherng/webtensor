import { Tensor } from '../../tensor';

/**
 * Return a view-like tensor with the gradient chain broken (no `_ctx`).
 * @category Memory
 */
export function detach(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: false,
  });
}
