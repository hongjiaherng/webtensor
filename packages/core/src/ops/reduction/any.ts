import { Tensor } from '../../tensor';
import { reduceOutputShape, normalizeAxes } from '../../shape';

/**
 * Logical OR over `axis` (or all axes if undefined). Input must be `bool`;
 * output is `bool`. IR op: `ReduceAny`. Not differentiable.
 * @category Reduction
 */
export function any(a: Tensor, axis?: number | number[], keepdim = false): Tensor {
  if (a.dtype !== 'bool') {
    throw new Error(`any: input dtype must be bool, got ${a.dtype}`);
  }
  const normalizedAxes = normalizeAxes(axis, a.shape.length);
  const outShape = reduceOutputShape(a.shape, normalizedAxes, keepdim);

  return new Tensor({
    shape: outShape,
    dtype: 'bool',
    device: a.device,
    requiresGrad: false,
    ctx: {
      op: 'ReduceAny',
      inputs: [a],
      attributes: { axes: normalizedAxes, keepdim: keepdim ? 1 : 0 },
    },
  });
}
