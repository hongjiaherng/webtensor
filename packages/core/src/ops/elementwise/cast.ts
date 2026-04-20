import { DType } from '@webtensor/ir';
import { Tensor } from '../../tensor';

/**
 * Element-wise dtype conversion:
 *   - float32 → int32:   truncates toward zero.
 *   - float32 → bool:    `x != 0.0` → true.
 *   - int32   → bool:    `x != 0`   → true.
 *   - bool    → int32:   false → 0, true → 1.
 *   - bool    → float32: false → 0.0, true → 1.0.
 *   - same dtype:        pure copy.
 *
 * Not differentiable — cast breaks the gradient chain. If you need a
 * differentiable variant, keep the tensor in its original dtype.
 */
export function cast(a: Tensor, dtype: DType): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype,
    device: a.device,
    requiresGrad: false, // cast is not differentiable
    ctx: {
      op: 'Cast',
      inputs: [a],
      attributes: { toDtype: dtype },
    },
  });
}
