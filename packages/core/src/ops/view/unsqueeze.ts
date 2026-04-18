import { Tensor } from '../../tensor';
import { squeeze } from './squeeze';

/** Insert a size-1 dimension at `dim`. */
export function unsqueeze(a: Tensor, dim: number): Tensor {
  const rank = a.shape.length;
  const d = dim < 0 ? rank + 1 + dim : dim;
  if (d < 0 || d > rank) {
    throw new Error(`unsqueeze: dim ${dim} out of range for tensor of rank ${rank}`);
  }
  const outShape = [...a.shape];
  outShape.splice(d, 0, 1);

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Unsqueeze',
      inputs: [a],
      attributes: { dim: d },
      backward: (grad) => [squeeze(grad, d)],
    },
  });
}
