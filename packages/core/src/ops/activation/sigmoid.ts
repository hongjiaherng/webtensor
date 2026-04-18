import { Tensor } from '../../tensor';
import { tensor } from '../../init/tensor';
import { mul } from '../elementwise/mul';
import { sub } from '../elementwise/sub';

/** Element-wise logistic `1 / (1 + exp(-a))`. */
export function sigmoid(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Sigmoid',
      inputs: [a],
      backward: (grad) => {
        const s = sigmoid(a);
        return [mul(grad, mul(s, sub(tensor([1]), s)))];
      },
    },
  });
}
