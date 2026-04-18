import { Tensor } from '../../tensor';
import { broadcastShapes } from '../../shape';
import { unbroadcastGrad } from '../_unbroadcast';
import { transpose } from '../view/transpose';
import { unsqueeze } from '../view/unsqueeze';
import { squeeze } from '../view/squeeze';

/**
 * PyTorch-style `matmul`:
 *   - 1D·1D → scalar (dot product)
 *   - 1D·2D → prepend 1 to a, compute, drop prepended dim
 *   - 2D·1D → append 1 to b, compute, drop appended dim
 *   - 2D·2D → matrix-matrix product
 *   - N-D   → batched matmul with broadcast over leading dims
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.shape.length === 0 || b.shape.length === 0) {
    throw new Error('MatMul requires inputs to be at least rank 1');
  }

  const aWas1D = a.shape.length === 1;
  const bWas1D = b.shape.length === 1;
  const aP = aWas1D ? unsqueeze(a, 0) : a;
  const bP = bWas1D ? unsqueeze(b, -1) : b;

  const rankA = aP.shape.length;
  const rankB = bP.shape.length;
  const K_A = aP.shape[rankA - 1];
  const K_B = bP.shape[rankB - 2];
  if (K_A !== null && K_B !== null && K_A !== K_B) {
    throw new Error(`MatMul inner dimensions must match: ${K_A} !== ${K_B}`);
  }

  const batchA = aP.shape.slice(0, -2);
  const batchB = bP.shape.slice(0, -2);
  const batchOut =
    batchA.length === 0 && batchB.length === 0 ? [] : broadcastShapes(batchA, batchB);

  const outShape = [...batchOut, aP.shape[rankA - 2], bP.shape[rankB - 1]];

  const raw = new Tensor({
    shape: outShape,
    dtype: aP.dtype,
    device: aP.device,
    requiresGrad: aP.requiresGrad || bP.requiresGrad,
    ctx: {
      op: 'MatMul',
      inputs: [aP, bP],
      backward: (grad) => {
        const gradA = matmul(grad, transpose(bP));
        const gradB = matmul(transpose(aP), grad);
        return [unbroadcastGrad(gradA, aP.shape), unbroadcastGrad(gradB, bP.shape)];
      },
    },
  });

  let out = raw;
  if (aWas1D) out = squeeze(out, -2);
  if (bWas1D) out = squeeze(out, -1);
  return out;
}
