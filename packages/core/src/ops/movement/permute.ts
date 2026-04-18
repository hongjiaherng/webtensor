import { Tensor } from '../../tensor';

/** Reorder axes according to `axes`. Zero-copy view. */
export function permute(a: Tensor, axes: number[]): Tensor {
  if (axes.length !== a.shape.length) {
    throw new Error(`permute: axes length ${axes.length} must match tensor rank ${a.shape.length}`);
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
