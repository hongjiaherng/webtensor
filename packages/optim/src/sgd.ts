import { TypedArray } from '@webtensor/runtime';
import { Tensor } from '@webtensor/core';
import { Optimizer } from './optimizer';

function dataOf(t: Tensor, label: string): TypedArray {
  if (!t.data) {
    throw new Error(
      `${label}: expected a tensor with .data (from a compile() output or a trainable param); got an unevaluated graph tensor.`,
    );
  }
  return t.data;
}

/**
 * Stochastic gradient descent with optional momentum.
 *
 * ```ts
 * const W = randn([2, 8], { requiresGrad: true });
 * const opt = new SGD(0.1, 0.9);   // lr, momentum
 *
 * for (let i = 0; i < 1000; i++) {
 *   const { loss, dW } = await step({ x, y });
 *   opt.step([W], [dW]);
 * }
 * ```
 */
export class SGD implements Optimizer {
  private lr: number;
  private momentum: number;
  private velocity: WeakMap<Tensor, Float32Array>;

  constructor(lr: number, momentum: number = 0) {
    this.lr = lr;
    this.momentum = momentum;
    this.velocity = new WeakMap();
  }

  step(params: Tensor[], grads: Tensor[]): void {
    if (params.length !== grads.length) {
      throw new Error(
        `SGD.step: params length ${params.length} !== grads length ${grads.length}`,
      );
    }
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      const pdata = dataOf(p, `SGD.step: param[${i}]`);
      const g = dataOf(grads[i], `SGD.step: grad[${i}]`);
      if (g.length !== pdata.length) {
        throw new Error(
          `SGD.step: grad[${i}] length ${g.length} !== param[${i}] length ${pdata.length}`,
        );
      }
      if (this.momentum > 0) {
        let v = this.velocity.get(p);
        if (!v || v.length !== pdata.length) {
          v = new Float32Array(pdata.length);
          this.velocity.set(p, v);
        }
        for (let j = 0; j < pdata.length; j++) {
          v[j] = this.momentum * v[j] + g[j];
          pdata[j] -= this.lr * v[j];
        }
      } else {
        for (let j = 0; j < pdata.length; j++) {
          pdata[j] -= this.lr * g[j];
        }
      }
    }
  }
}
