import { Tensor } from '../tensor';
import { shapesEqual } from '../shape';
import { sum } from './reduce/sum';
import { reshape } from './view/reshape';

/**
 * Internal helper: reduce a gradient back to a target shape by summing over
 * any axes that were broadcast during the forward pass. Used by every binary
 * op backward closure and by `Expand` backward. Not exported from the package.
 */
export function unbroadcastGrad(grad: Tensor, targetShape: (number | null)[]): Tensor {
  const gradShape = grad.shape;
  if (shapesEqual(gradShape, targetShape)) return grad;

  const gradRank = gradShape.length;
  const targetRank = targetShape.length;
  const rankDiff = gradRank - targetRank;
  const sumAxes: number[] = [];

  // Leading axes prepended by broadcasting
  for (let i = 0; i < rankDiff; i++) sumAxes.push(i);

  // Axes where target is size-1 but grad is larger
  for (let i = 0; i < targetRank; i++) {
    if (targetShape[i] === 1 && (gradShape[rankDiff + i] ?? 1) !== 1) {
      sumAxes.push(rankDiff + i);
    }
  }

  let g = grad;
  if (sumAxes.length > 0) g = sum(g, sumAxes, /* keepdim */ true);
  if (!shapesEqual(g.shape, targetShape)) g = reshape(g, targetShape as number[]);
  return g;
}
