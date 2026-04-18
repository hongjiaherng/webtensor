import { Tensor } from '../../tensor';
import { div } from '../binary/div';
import { mul } from '../binary/mul';
import { tensor } from '../../init/tensor';

/** Element-wise `sqrt(a)`. */
export function sqrt(a: Tensor): Tensor {
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
