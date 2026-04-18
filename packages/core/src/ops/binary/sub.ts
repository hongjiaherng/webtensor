import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { unbroadcastGrad } from '../_unbroadcast';
import { neg } from '../unary/neg';

/** Element-wise `a - b` with broadcasting. */
export function sub(a: Tensor, b: Tensor): Tensor {
  return new Tensor({
    shape: broadcastShapes(a.shape, b.shape),
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad || b.requiresGrad,
    ctx: {
      op: 'Sub',
      inputs: [a, b],
      backward: (grad) => [
        unbroadcastGrad(grad, a.shape),
        unbroadcastGrad(neg(grad), b.shape),
      ],
    },
  });
}
