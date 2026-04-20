import { Tensor } from '../../tensor';

/**
 * Reorder axes according to `axes`. Zero-copy view.
 * @category Movement
 */
export function permute(a: Tensor, axes: number[]): Tensor {
  const rank = a.shape.length;
  if (axes.length !== rank) {
    throw new Error(`permute: axes length ${axes.length} must match tensor rank ${rank}`);
  }
  const seen = new Array<boolean>(rank).fill(false);
  for (const ax of axes) {
    if (!Number.isInteger(ax) || ax < 0 || ax >= rank) {
      throw new Error(`permute: axis ${ax} out of range for rank ${rank}`);
    }
    if (seen[ax]) {
      throw new Error(`permute: axes must be a permutation; axis ${ax} repeated`);
    }
    seen[ax] = true;
  }
  const outShape = axes.map((ax) => a.shape[ax]);
  const inverseAxes = new Array(axes.length);
  for (let i = 0; i < axes.length; i++) inverseAxes[axes[i]] = i;

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Permute',
      inputs: [a],
      attributes: { axes },
      backward: (grad) => [permute(grad, inverseAxes)],
    },
  });
}
