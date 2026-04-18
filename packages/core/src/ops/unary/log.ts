import { Tensor } from '../../tensor';
import { div } from '../binary/div';

/** Element-wise natural log `log(a)`. */
export function log(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Log',
      inputs: [a],
      backward: (grad) => [div(grad, a)],
    },
  });
}
