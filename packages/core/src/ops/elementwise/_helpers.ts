import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { isArithmeticDType } from '@webtensor/runtime';

/**
 * Factory for element-wise comparison ops (eq / ne / lt / le / gt / ge).
 *
 * All comparisons produce a `bool` output with the broadcast shape of the
 * inputs. They are not differentiable (bool output).
 *
 * Inputs must be the same arithmetic dtype (float32 or int32). Cross-dtype
 * comparisons require an explicit `cast()` — this keeps kernel variants
 * bounded at one-per-dtype instead of one-per-pair.
 */
export function makeCompareOp(opName: string): (a: Tensor, b: Tensor) => Tensor {
  return (a: Tensor, b: Tensor): Tensor => {
    if (!isArithmeticDType(a.dtype) || !isArithmeticDType(b.dtype)) {
      throw new Error(
        `${opName.toLowerCase()}: bool is not a valid comparison input — cast to int32 or float32 first`,
      );
    }
    if (a.dtype !== b.dtype) {
      throw new Error(
        `${opName.toLowerCase()}: input dtypes ${a.dtype} and ${b.dtype} differ; cast one side first`,
      );
    }
    return new Tensor({
      shape: broadcastShapes(a.shape, b.shape),
      dtype: 'bool',
      device: a.device,
      requiresGrad: false,
      ctx: { op: opName, inputs: [a, b] },
    });
  };
}
