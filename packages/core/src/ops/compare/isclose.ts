import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';

export interface IsCloseOptions {
  /** Relative tolerance. Default `1e-5`. */
  rtol?: number;
  /** Absolute tolerance. Default `1e-8`. */
  atol?: number;
  /** If true, NaN on both sides compares equal. Default `false`. */
  equalNan?: boolean;
}

/**
 * Element-wise `|a - b| <= atol + rtol * |b|`, broadcasting.
 *
 * Float32-only. For exact equality on integers or bool, use `eq`. Returns a
 * non-differentiable bool tensor — matches PyTorch's `torch.isclose`.
 */
export function isclose(a: Tensor, b: Tensor, opts: IsCloseOptions = {}): Tensor {
  if (a.dtype !== 'float32' || b.dtype !== 'float32') {
    throw new Error(
      `isclose: requires float32 inputs; got ${a.dtype}/${b.dtype}. Use 'eq' for exact equality on int32/bool`,
    );
  }
  const rtol = opts.rtol ?? 1e-5;
  const atol = opts.atol ?? 1e-8;
  const equalNan = opts.equalNan ?? false;

  return new Tensor({
    shape: broadcastShapes(a.shape, b.shape),
    dtype: 'bool',
    device: a.device,
    requiresGrad: false,
    ctx: {
      op: 'IsClose',
      inputs: [a, b],
      attributes: { rtol, atol, equalNan: equalNan ? 1 : 0 },
    },
  });
}
