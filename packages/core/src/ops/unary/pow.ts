import { Tensor } from '../../tensor';
import { mul } from '../binary/mul';
import { tensor } from '../../init/tensor';

/** Element-wise `a^exponent`. `exponent` is a scalar. */
export function pow(a: Tensor, exponent: number): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Pow',
      inputs: [a],
      attributes: { exponent },
      backward: (grad) => [mul(mul(grad, tensor([exponent])), pow(a, exponent - 1))],
    },
  });
}
