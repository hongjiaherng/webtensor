import { Tensor } from '../tensor';
import { InitOptions, buildFromBuffer, makeTypedArray, totalElements } from './_internal';

/**
 * All-one tensor. Mirrors `torch.ones`.
 * @category Factories
 */
export function ones(shape: (number | null)[], options?: InitOptions): Tensor {
  const dtype = options?.dtype ?? 'float32';
  const n = totalElements(shape);
  const buffer = makeTypedArray(dtype, n);
  for (let i = 0; i < n; i++) buffer[i] = 1;
  return buildFromBuffer(shape, buffer, options);
}
