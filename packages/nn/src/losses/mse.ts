import { Tensor, sub, pow, mean } from '@webtensor/core';

/**
 * Mean-squared error: `mean((pred - target)^2)`. Returns a scalar tensor.
 */
export function mseLoss(pred: Tensor, target: Tensor): Tensor {
  return mean(pow(sub(pred, target), 2));
}
