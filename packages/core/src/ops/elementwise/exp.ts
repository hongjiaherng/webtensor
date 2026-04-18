import { Tensor } from '../../tensor';
import { mul } from './mul';

/** Element-wise `exp(a)`. Float-only — cast int32 → float32 first. */
export function exp(a: Tensor): Tensor {
  if (a.dtype !== 'float32') {
    throw new Error(`exp: requires float32 input, got ${a.dtype}. Cast first.`);
  }
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Exp',
      inputs: [a],
      backward: (grad) => [mul(grad, exp(a))],
    },
  });
}
