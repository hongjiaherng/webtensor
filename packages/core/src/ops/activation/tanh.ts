import { Tensor } from '../../tensor';
import { tensor } from '../../init/tensor';
import { mul } from '../elementwise/mul';
import { sub } from '../elementwise/sub';

/** Element-wise `tanh(a)`. */
export function tanh(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Tanh',
      inputs: [a],
      backward: (grad) => {
        const t = tanh(a);
        return [mul(grad, sub(tensor([1]), mul(t, t)))];
      },
    },
  });
}
