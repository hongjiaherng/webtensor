import { resultDType, isArithmeticDType } from '@webtensor/runtime';
import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { sumToShape } from '../../autograd/sumToShape';

/** Element-wise `a * b` with broadcasting and dtype promotion. */
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
      backward: (grad) => [sumToShape(mul(grad, b), a.shape), sumToShape(mul(grad, a), b.shape)],
    },
  });
}
