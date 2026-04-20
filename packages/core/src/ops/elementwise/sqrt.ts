import { Tensor } from '../../tensor';
import { div } from './div';
import { mul } from './mul';
import { tensor } from '../../init/tensor';

/**
 * Element-wise `sqrt(a)`. Float-only — cast int32 → float32 first.
 * @category Elementwise
 */
export function sqrt(a: Tensor): Tensor {
  if (a.dtype !== 'float32') {
    throw new Error(`sqrt: requires float32 input, got ${a.dtype}. Cast first.`);
  }
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Sqrt',
      inputs: [a],
      backward: (grad) => [div(grad, mul(tensor([2]), sqrt(a)))],
    },
  });
}
