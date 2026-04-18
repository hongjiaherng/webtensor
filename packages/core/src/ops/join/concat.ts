import { Tensor } from '../../tensor';
import { slice } from '../view/slice';

/**
 * Concatenate tensors along an existing axis. ONNX: `Concat`. Matches
 * `torch.cat` / `numpy.concatenate` semantics — no broadcasting, all inputs
 * must agree on every non-axis dimension and share a dtype.
 *
 * Backward: each input's gradient is the corresponding axis-slice of the
 * output gradient. The slice is a zero-copy view, so `concat` adds no
 * copy-back cost beyond what the user's downstream ops require.
 */
export function concat(tensors: Tensor[], axis: number = 0): Tensor {
  if (tensors.length === 0) throw new Error('concat: need at least one tensor');
  const first = tensors[0];
  const rank = first.shape.length;

  // Normalize axis: allow negative indices (PyTorch / NumPy parity).
  const ax = axis < 0 ? axis + rank : axis;
  if (ax < 0 || ax >= rank) {
    throw new Error(`concat: axis ${axis} out of range for rank ${rank}`);
  }

  // Validate: all tensors share rank, dtype, and every non-axis dim.
  for (let t = 1; t < tensors.length; t++) {
    const s = tensors[t].shape;
    if (s.length !== rank) {
      throw new Error(`concat: input ${t} has rank ${s.length}, expected ${rank}`);
    }
    if (tensors[t].dtype !== first.dtype) {
      throw new Error(
        `concat: input ${t} has dtype ${tensors[t].dtype}, expected ${first.dtype}; cast first`,
      );
    }
    for (let d = 0; d < rank; d++) {
      if (d === ax) continue;
      if (s[d] !== first.shape[d]) {
        throw new Error(
          `concat: input ${t} dim ${d} is ${s[d]}, expected ${first.shape[d]} (axis=${ax})`,
        );
      }
    }
  }

  // Output shape: inherit all dims; replace axis with the sum.
  const outShape = first.shape.slice();
  outShape[ax] = tensors.reduce((sum, t) => sum + (t.shape[ax] as number), 0);

  const requiresGrad = tensors.some((t) => t.requiresGrad);

  return new Tensor({
    shape: outShape,
    dtype: first.dtype,
    device: first.device,
    requiresGrad,
    ctx: {
      op: 'Concat',
      inputs: tensors,
      attributes: { axis: ax },
      // Splits the output grad back into per-input slices along `axis`.
      backward: (grad) => {
        const grads: Tensor[] = [];
        let start = 0;
        for (const t of tensors) {
          const size = t.shape[ax] as number;
          const starts = outShape.map(() => 0);
          const ends = outShape.map((d) => d as number);
          starts[ax] = start;
          ends[ax] = start + size;
          grads.push(slice(grad, starts, ends));
          start += size;
        }
        return grads;
      },
    },
  });
}
