import { Tensor } from '../tensor';
import { InitOptions } from './_internal';
import { zeros } from './zeros';
import { ones } from './ones';
import { randn } from './randn';

/** Zero tensor matching `t`'s shape and dtype. */
export function zerosLike(t: Tensor, options?: InitOptions): Tensor {
  return zeros(t.shape, { dtype: t.dtype, device: t.device, ...options });
}

/** One tensor matching `t`'s shape and dtype. */
export function onesLike(t: Tensor, options?: InitOptions): Tensor {
  return ones(t.shape, { dtype: t.dtype, device: t.device, ...options });
}

/** Standard-normal tensor matching `t`'s shape. */
export function randnLike(
  t: Tensor,
  options?: InitOptions & { seed?: number; mean?: number; std?: number },
): Tensor {
  return randn(t.shape, { dtype: t.dtype, device: t.device, ...options });
}
