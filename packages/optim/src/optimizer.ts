import { Tensor } from '@webtensor/core';

/**
 * Minimal optimizer interface. Concrete optimizers mutate `param.data` in
 * place using `grad.data` from each compiled training step.
 */
export interface Optimizer {
  step(params: Tensor[], grads: Tensor[]): void;
}
