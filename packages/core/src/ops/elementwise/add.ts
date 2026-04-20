import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { sumToShape } from '../../autograd/sumToShape';
import { resultDType, isArithmeticDType } from '@webtensor/runtime';

/**
 * Element-wise `a + b` with broadcasting and dtype promotion.
 *
 * @example
 * ```ts
 * import { tensor, add, run } from '@webtensor/core';
 *
 * const a = tensor([[1, 2], [3, 4]]);
 * const b = tensor([10, 20]);              // broadcasts across rows
 * const y = await run(add(a, b));          // [[11, 22], [13, 24]]
 * ```
 *
 * @category Elementwise
 */
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
