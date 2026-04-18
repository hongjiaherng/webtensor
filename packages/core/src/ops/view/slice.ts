import { Tensor } from '../../tensor';

/**
 * Slice each dim to `[starts[i], ends[i])`. Zero-copy view.
 *
 * NOTE: backward is not implemented (would require a `Pad`/`Scatter` op to
 * place the gradient into zeros at the slice region).
 */
export function slice(a: Tensor, starts: number[], ends: number[]): Tensor {
  if (starts.length !== a.shape.length || ends.length !== a.shape.length) {
    throw new Error('slice: starts and ends must have the same length as the tensor rank');
  }
  for (let i = 0; i < starts.length; i++) {
    const dim = a.shape[i];
    if (starts[i] < 0 || (dim !== null && ends[i] > dim) || starts[i] >= ends[i]) {
      throw new Error(`slice: invalid range [${starts[i]}, ${ends[i]}) for dim ${i} (size ${dim})`);
    }
  }
  const outShape = starts.map((s, i) => ends[i] - s);

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Slice',
      inputs: [a],
      attributes: { starts, ends },
    },
  });
}
