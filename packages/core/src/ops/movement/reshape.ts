import { Tensor } from '../../tensor';
import { resolveShapeInference } from '../../shape';

/**
 * Reshape to `shape`. Auto-copies if the source is non-contiguous.
 * A single `null` or `-1` in `shape` is inferred from the input's total size.
 * @category Movement
 */
export function reshape(a: Tensor, shape: (number | null)[]): Tensor {
  const resolved = resolveShapeInference(a.shape, shape);
  return new Tensor({
    shape: resolved,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Reshape',
      inputs: [a],
      attributes: { shape: resolved },
      backward: (grad) => [reshape(grad, a.shape)],
    },
  });
}
