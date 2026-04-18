import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { unbroadcastGrad } from '../_unbroadcast';

/** Element-wise `a * b` with broadcasting. */
export function mul(a: Tensor, b: Tensor): Tensor {
  return new Tensor({
    shape: broadcastShapes(a.shape, b.shape),
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad || b.requiresGrad,
    ctx: {
      op: 'Mul',
      inputs: [a, b],
      backward: (grad) => [
        unbroadcastGrad(mul(grad, b), a.shape),
        unbroadcastGrad(mul(grad, a), b.shape),
      ],
    },
  });
}
