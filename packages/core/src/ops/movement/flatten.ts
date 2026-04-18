import { Tensor } from '../../tensor';
import { reshape } from './reshape';

/** Flatten dims `[startDim, endDim]` (inclusive) into one. */
export function flatten(a: Tensor, startDim = 0, endDim = -1): Tensor {
  const rank = a.shape.length;
  const sd = startDim < 0 ? rank + startDim : startDim;
  const ed = endDim < 0 ? rank + endDim : endDim;
  if (sd < 0 || sd >= rank || ed < 0 || ed >= rank || sd > ed) {
    throw new Error(`flatten: invalid dims [${startDim}, ${endDim}] for rank ${rank}`);
  }
  const before = a.shape.slice(0, sd) as number[];
  const middle = a.shape.slice(sd, ed + 1) as number[];
  const after = a.shape.slice(ed + 1) as number[];
  const flatDim = middle.reduce((acc, d) => acc * (d ?? 1), 1);
  return reshape(a, [...before, flatDim, ...after]);
}
