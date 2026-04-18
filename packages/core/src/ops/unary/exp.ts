import { Tensor } from '../../tensor';
import { mul } from '../binary/mul';

/** Element-wise `exp(a)`. */
export function exp(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Exp',
      inputs: [a],
      backward: (grad) => [mul(grad, exp(a))],
    },
  });
}
