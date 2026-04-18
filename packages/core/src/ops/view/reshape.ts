import { Tensor } from '../../tensor';

/** Reshape to `shape`. Auto-copies if the source is non-contiguous (PyTorch semantics). */
export function reshape(a: Tensor, shape: number[]): Tensor {
  return new Tensor({
    shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Reshape',
      inputs: [a],
      attributes: { shape },
      backward: (grad) => [reshape(grad, a.shape as number[])],
    },
  });
}
