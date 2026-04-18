import { DType } from '@webtensor/ir';
import { Tensor } from '../tensor';
import { Device } from '../types';

/**
 * Low-level primitive: create a `Placeholder` tensor with no attached data.
 * Data is supplied at `engine.evaluate(graph, feeds)` time via the `feeds` map
 * keyed on the returned tensor's `id`.
 *
 * In most day-to-day training code you shouldn't need this. The high-level
 * `compile(fn, spec)` helper (in `./compile.ts`) creates placeholders for you
 * and hides the `.id` bookkeeping.
 */
export function placeholder(
  shape: number[],
  dtype: DType = 'float32',
  device: Device = 'cpu',
): Tensor {
  return new Tensor({
    shape,
    dtype,
    device,
    requiresGrad: false,
    ctx: {
      op: 'Placeholder',
      inputs: [],
      attributes: { shape },
    },
  });
}
