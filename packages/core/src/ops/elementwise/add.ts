import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { sumToShape } from '../../autograd/sumToShape';
import { resultDType, isArithmeticDType } from '@webtensor/runtime';

/** Element-wise `a + b` with broadcasting and dtype promotion. */
export function add(a: Tensor, b: Tensor): Tensor {
  if (!isArithmeticDType(a.dtype) || !isArithmeticDType(b.dtype)) {
    throw new Error(
      `add: dtype ${a.dtype}/${b.dtype} not supported for arithmetic; cast to int32 or float32 first`,
    );
  }
  return new Tensor({
    shape: broadcastShapes(a.shape, b.shape),
    dtype: resultDType(a.dtype, b.dtype),
    device: a.device,
    requiresGrad: a.requiresGrad || b.requiresGrad,
    ctx: {
      op: 'Add',
      inputs: [a, b],
      backward: (grad) => [sumToShape(grad, a.shape), sumToShape(grad, b.shape)],
    },
  });
}
