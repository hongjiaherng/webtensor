import { Tensor } from '../../tensor';
import { reduceOutputShape, normalizeAxes } from '../../shape';
import { reshape } from '../movement/reshape';
import { expand } from '../movement/expand';
import { contiguous } from '../memory/contiguous';
import { mul } from '../elementwise/mul';
import { tensor } from '../../init/tensor';

/** Mean over `axis` (or all axes if undefined). IR op: `ReduceMean`. */
export function mean(a: Tensor, axis?: number | number[], keepdim = false): Tensor {
  const normalizedAxes = normalizeAxes(axis, a.shape.length);
  const outShape = reduceOutputShape(a.shape, normalizedAxes, keepdim);
  const reduceSize = normalizedAxes.reduce((acc, ax) => acc * (a.shape[ax] as number), 1);
  const axisSet = new Set(normalizedAxes);
  const keepdimShape = a.shape.map((d, i) => (axisSet.has(i) ? 1 : (d ?? 1)));

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'ReduceMean',
      inputs: [a],
      attributes: { axes: normalizedAxes, keepdim: keepdim ? 1 : 0 },
      backward: (grad) => {
        let g = grad;
        if (!keepdim) g = reshape(g, keepdimShape);
        return [mul(contiguous(expand(g, a.shape as number[])), tensor([1 / reduceSize]))];
      },
    },
  });
}

/** @deprecated Use `mean`. */
export const reduceMean = mean;
