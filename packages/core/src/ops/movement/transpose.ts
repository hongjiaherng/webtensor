import { Tensor } from '../../tensor';

/**
 * Swap the last two dimensions. Zero-copy view.
 * @category Movement
 */
export function transpose(a: Tensor): Tensor {
  if (a.shape.length < 2) {
    throw new Error('Transpose requires at least 2 dimensions');
  }
  const outShape = [...a.shape];
  const last = outShape.length - 1;
  [outShape[last - 1], outShape[last]] = [outShape[last], outShape[last - 1]];

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Transpose',
      inputs: [a],
      backward: (grad) => [transpose(grad)],
    },
  });
}
