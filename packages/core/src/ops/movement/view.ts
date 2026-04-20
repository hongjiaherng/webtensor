import { Tensor } from '../../tensor';
import { resolveShapeInference } from '../../shape';

/**
 * Strict reshape: throws on non-contiguous input. Use `.contiguous()` first
 * or prefer `reshape()` (which auto-copies). A single `null` in `shape` is
 * or `-1` is inferred from the input's total size (PyTorch semantics).
 */
export function view(a: Tensor, shape: (number | null)[]): Tensor {
  const resolved = resolveShapeInference(a.shape, shape);
  return new Tensor({
    shape: resolved,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'View',
      inputs: [a],
      attributes: { shape: resolved },
      backward: (grad) => [view(grad, a.shape)],
    },
  });
}
