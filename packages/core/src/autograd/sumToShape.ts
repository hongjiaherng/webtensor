import { Tensor } from '../tensor';
import { shapesEqual } from '../shape';
import { sum } from '../ops/reduction/sum';
import { reshape } from '../ops/movement/reshape';

/**
 * Reduce a gradient tensor back to `targetShape` by summing over axes that
 * were broadcast during the forward pass. Used by binary-op backward closures
 * and by `Expand` backward.
 * @category Autograd
 */
export function sumToShape(grad: Tensor, targetShape: (number | null)[]): Tensor {
  const gradShape = grad.shape;
  if (shapesEqual(gradShape, targetShape)) return grad;

  const gradRank = gradShape.length;
  const targetRank = targetShape.length;
  const rankDiff = gradRank - targetRank;
  const sumAxes: number[] = [];

  for (let i = 0; i < rankDiff; i++) sumAxes.push(i);
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
