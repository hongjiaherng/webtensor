import { resultDType, isArithmeticDType } from '@webtensor/runtime';
import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { unbroadcastGrad } from '../_unbroadcast';

/** Element-wise `a * b` with broadcasting. PyTorch-style dtype promotion. */
export function mul(a: Tensor, b: Tensor): Tensor {
  if (!isArithmeticDType(a.dtype) || !isArithmeticDType(b.dtype)) {
    throw new Error(
      `mul: dtype ${a.dtype}/${b.dtype} not supported for arithmetic; cast to int32 or float32 first`,
    );
  }
  return new Tensor({
    shape: broadcastShapes(a.shape, b.shape),
    dtype: resultDType(a.dtype, b.dtype),
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
