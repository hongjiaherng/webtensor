import { resultDType, isArithmeticDType } from '@webtensor/runtime';
import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { sumToShape } from '../../autograd/sumToShape';
import { mul } from './mul';
import { tensor } from '../../init/tensor';

/** Element-wise `a / b` with broadcasting and dtype promotion. */
export function div(a: Tensor, b: Tensor): Tensor {
  if (!isArithmeticDType(a.dtype) || !isArithmeticDType(b.dtype)) {
    throw new Error(
      `div: dtype ${a.dtype}/${b.dtype} not supported for arithmetic; cast to int32 or float32 first`,
    );
  }
  return new Tensor({
    shape: broadcastShapes(a.shape, b.shape),
    dtype: resultDType(a.dtype, b.dtype),
    device: a.device,
    requiresGrad: a.requiresGrad || b.requiresGrad,
    ctx: {
      op: 'Div',
      inputs: [a, b],
      backward: (grad) => {
        const gradA = div(grad, b);
        const gradB = mul(div(mul(grad, a), mul(b, b)), tensor([-1]));
        return [sumToShape(gradA, a.shape), sumToShape(gradB, b.shape)];
      },
    },
  });
}
