import { Tensor } from '../../tensor';
import { sumToShape } from '../../autograd/sumToShape';

/**
 * Expand size-1 dims to `shape`. Zero-copy stride-0 broadcast.
 * @category Movement
 */
export function expand(a: Tensor, shape: number[]): Tensor {
  if (shape.length < a.shape.length) {
    throw new Error(
      `expand: target shape rank ${shape.length} must be >= tensor rank ${a.shape.length}`,
    );
  }
  const rankOffset = shape.length - a.shape.length;
  for (let i = 0; i < a.shape.length; i++) {
    const srcDim = a.shape[i];
    const tgtDim = shape[rankOffset + i];
    if (srcDim !== 1 && srcDim !== tgtDim) {
      throw new Error(
        `expand: cannot expand dim ${i} from ${srcDim} to ${tgtDim} (must be 1 or match)`,
      );
    }
  }

  return new Tensor({
    shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Expand',
      inputs: [a],
      attributes: { shape },
      backward: (grad) => [sumToShape(grad, a.shape)],
    },
  });
}
