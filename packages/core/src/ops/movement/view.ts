import { Tensor } from '../../tensor';

/**
 * Strict reshape: throws on non-contiguous input. Use `.contiguous()` first
 * or prefer `reshape()` (which auto-copies).
 */
export function view(a: Tensor, shape: (number | null)[]): Tensor {
  return new Tensor({
    shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'View',
      inputs: [a],
      attributes: { shape },
      backward: (grad) => [view(grad, a.shape)],
    },
  });
}
