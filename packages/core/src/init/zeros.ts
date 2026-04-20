import { Tensor } from '../tensor';
import { InitOptions, buildFromBuffer, makeTypedArray, totalElements } from './_internal';

/**
 * All-zero tensor. Mirrors `torch.zeros`.
 * @category Factories
 */
export function zeros(shape: (number | null)[], options?: InitOptions): Tensor {
  const dtype = options?.dtype ?? 'float32';
  const buffer = makeTypedArray(dtype, totalElements(shape));
  return buildFromBuffer(shape, buffer, options);
}
