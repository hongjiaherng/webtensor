import { Tensor } from '../../tensor';
import { reshape } from './reshape';

/**
 * Remove a size-1 dim at `dim`, or all size-1 dims if `dim` is omitted.
 */
export function squeeze(a: Tensor, dim?: number): Tensor {
  let outShape: (number | null)[];
  if (dim !== undefined) {
    const d = dim < 0 ? a.shape.length + dim : dim;
    if (d < 0 || d >= a.shape.length) {
      throw new Error(`squeeze: dim ${dim} out of range for rank ${a.shape.length}`);
    }
    if (a.shape[d] !== 1) {
      throw new Error(`squeeze: dimension ${d} has size ${a.shape[d]}, expected 1`);
    }
    outShape = [...a.shape];
    outShape.splice(d, 1);
  } else {
    outShape = a.shape.filter((s) => s !== 1);
    if (outShape.length === 0) outShape = [1];
  }

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Squeeze',
      inputs: [a],
      attributes: dim !== undefined ? { dim: dim < 0 ? a.shape.length + dim : dim } : undefined,
      backward: (grad) => [reshape(grad, a.shape)],
    },
  });
}
