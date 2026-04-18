import { Tensor } from '../../tensor';
import { div } from './div';

/** Element-wise natural log `log(a)`. Float-only — cast int32 → float32 first. */
export function log(a: Tensor): Tensor {
  if (a.dtype !== 'float32') {
    throw new Error(`log: requires float32 input, got ${a.dtype}. Cast first.`);
  }
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Log',
      inputs: [a],
      backward: (grad) => [div(grad, a)],
    },
  });
}
