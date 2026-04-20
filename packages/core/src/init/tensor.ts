import { typedArrayCtor, TypedArray } from '@webtensor/runtime';
import { Tensor } from '../tensor';
import { NestedArray } from '../types';
import { inferShape, flattenArray } from '../shape';
import { InitOptions, buildFromBuffer } from './_internal';

/**
 * Create a tensor from a nested array of numbers.
 *
 * @example
 * ```ts
 * import { tensor } from '@webtensor/core';
 *
 * const x = tensor([[1, 2], [3, 4]]);                      // 2×2 Constant
 * const w = tensor([0.1, 0.2], { requiresGrad: true });    // trainable 1D param
 * const y = tensor([1, 2, 3], { dtype: 'int32' });         // int32 vector
 * ```
 *
 * @category Factories
 */
export function tensor(data: NestedArray<number>, options?: InitOptions): Tensor {
  const extractedShape = inferShape(data);
  // Bare number → rank-0 scalar. `inferShape(42)` is `[]`.
  const shape = options?.shape ?? extractedShape;
  const flattened = flattenArray(data);

  const dtype = options?.dtype ?? 'float32';
  const Ctor = typedArrayCtor(dtype);
  const buffer = new Ctor(flattened) as TypedArray;

  return buildFromBuffer(shape, buffer, options);
}
