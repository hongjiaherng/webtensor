import { Tensor } from '../../tensor';
import { slice } from './slice';
import { contiguous } from '../memory/contiguous';

/**
 * Constant-value padding. ONNX: `Pad` (mode="constant"). Matches
 * `torch.nn.functional.pad` and `numpy.pad` semantics for constant padding.
 *
 * `pads` has length `2 * rank` packed as
 * `[before_0, ..., before_{rank-1}, after_0, ..., after_{rank-1}]` — the same
 * layout ONNX uses. Output dim `d` is `input.shape[d] + pads[d] + pads[rank+d]`.
 *
 * Backward: `grad_input = slice(grad_out, pads_before, pads_before + input_shape)`.
 * `value` carries no gradient (it's a constant).
 *
 * Note: ONNX Pad opset ≥ 11 takes `pads` as an input tensor. We keep it as an
 * attribute for compactness; ONNX export will hoist it to an initializer.
 */
export function pad(a: Tensor, pads: number[], value: number = 0): Tensor {
  const rank = a.shape.length;
  if (pads.length !== 2 * rank) {
    throw new Error(`pad: pads length ${pads.length} must equal 2 * rank (${2 * rank})`);
  }
  for (let i = 0; i < pads.length; i++) {
    if (pads[i] < 0) throw new Error(`pad: negative pad at index ${i} not supported`);
  }
  const padsBefore = pads.slice(0, rank);
  const padsAfter = pads.slice(rank);
  const inputShape = a.shape.map((d) => d as number);
  const outShape = inputShape.map((d, i) => d + padsBefore[i] + padsAfter[i]);

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Pad',
      inputs: [a],
      attributes: { pads, value },
      backward: (grad) => {
        const starts = padsBefore.slice();
        const ends = padsBefore.map((b, i) => b + inputShape[i]);
        return [contiguous(slice(grad, starts, ends))];
      },
    },
  });
}
