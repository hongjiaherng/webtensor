import { Tensor } from '../../tensor';
import { reduceOutputShape, normalizeAxes } from '../../shape';
import { reshape } from '../movement/reshape';
import { expand } from '../movement/expand';
import { contiguous } from '../memory/contiguous';

/**
 * Sum over `axis` (or all axes if undefined). IR op: `ReduceSum`.
 *
 * `axis` accepts a single int, an array, or undefined (reduce-all).
 * Negative indices count from the end.
 * @category Reduction
 */
export function sum(a: Tensor, axis?: number | number[], keepdim = false): Tensor {
  const normalizedAxes = normalizeAxes(axis, a.shape.length);
  const outShape = reduceOutputShape(a.shape, normalizedAxes, keepdim);
  const axisSet = new Set(normalizedAxes);
  const keepdimShape = a.shape.map((d, i) => (axisSet.has(i) ? 1 : (d ?? 1)));

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'ReduceSum',
      inputs: [a],
      attributes: { axes: normalizedAxes, keepdim: keepdim ? 1 : 0 },
      backward: (grad) => {
        let g = grad;
        if (!keepdim) g = reshape(g, keepdimShape);
        return [contiguous(expand(g, a.shape as number[]))];
      },
    },
  });
}
