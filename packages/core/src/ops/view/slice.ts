import { Tensor } from '../../tensor';
import { pad } from '../padding/pad';

/**
 * Slice each dim to `[starts[i], ends[i])`. Zero-copy view.
 *
 * Backward: scatter the gradient back into a zero tensor at the sliced region.
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
  const parentShape = a.shape.map((d) => d as number);
  const rank = parentShape.length;

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Slice',
      inputs: [a],
      attributes: { starts, ends },
      // grad of slice = pad(grad_out, before=starts, after=parentShape-ends).
      backward: (grad) => {
        const pads: number[] = new Array(2 * rank);
        for (let d = 0; d < rank; d++) {
          pads[d] = starts[d];
          pads[rank + d] = parentShape[d] - ends[d];
        }
        return [pad(grad, pads, 0)];
      },
    },
  });
}
