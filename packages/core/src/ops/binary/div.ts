import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { unbroadcastGrad } from '../_unbroadcast';
import { mul } from './mul';
import { tensor } from '../../init/tensor';

/** Element-wise `a / b` with broadcasting. */
export function div(a: Tensor, b: Tensor): Tensor {
  return new Tensor({
    shape: broadcastShapes(a.shape, b.shape),
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad || b.requiresGrad,
    ctx: {
      op: 'Div',
      inputs: [a, b],
      backward: (grad) => {
        const gradA = div(grad, b);
        const gradB = mul(div(mul(grad, a), mul(b, b)), tensor([-1]));
        return [unbroadcastGrad(gradA, a.shape), unbroadcastGrad(gradB, b.shape)];
      },
    },
  });
}
