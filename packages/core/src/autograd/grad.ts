import { Tensor } from '../tensor';

const backwardDone = new WeakSet<Tensor>();

/**
 * Returns the gradient of `loss` with respect to `param` as an unevaluated
 * `Tensor`. Intended for use inside a `compile()`-traced function alongside
 * the loss itself.
 *
 * The idiomatic pattern: create parameters with `requiresGrad: true` outside
 * the traced function; `compile()` auto-feeds them on every call. Users don't
 * touch `Float32Array`s directly.
 *
 * Call multiple times with the same `loss` to get gradients for different
 * params — `loss.backward()` is only run once per loss tensor.
 *
 * @example
 * ```ts
 * import { randn, grad, compile } from '@webtensor/core';
 * import { mseLoss } from '@webtensor/nn';
 *
 * const W = randn([2, 4], { requiresGrad: true });
 * const step = compile((x, y) => {
 *   const yhat = x.matmul(W);
 *   const loss = mseLoss(yhat, y);
 *   return { loss, gW: grad(loss, W) };
 * }, { x: [null, 2], y: [null, 4] });
 * ```
 *
 * @category Autograd
 */
export function grad(loss: Tensor, param: Tensor): Tensor {
  if (!loss.requiresGrad) {
    throw new Error(
      'grad: loss does not depend on any tensor with requiresGrad=true. ' +
        'Make sure at least one `randn(..., { requiresGrad: true })` param feeds into the loss.',
    );
  }
  if (!backwardDone.has(loss)) {
    loss.backward();
    backwardDone.add(loss);
  }
  if (!param.grad) {
    throw new Error('grad: param has no gradient — it is not part of the loss computation graph.');
  }
  return param.grad;
}
