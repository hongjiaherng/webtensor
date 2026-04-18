import { typedArrayCtor, TypedArray } from '@webtensor/runtime';
import { Tensor } from '../tensor';
import { NestedArray } from '../types';
import { inferShape, flattenArray } from '../shape';
import { InitOptions, buildFromBuffer } from './_internal';

/**
 * Create a tensor from a nested array of numbers. Mirrors `torch.tensor([...])`.
 *
 * ```ts
 * const x = tensor([[1, 2], [3, 4]]);                      // 2×2 Constant
 * const w = tensor([0.1, 0.2], { requiresGrad: true });    // trainable 1D param
 * ```
 */
export function tensor(data: NestedArray<number>, options?: InitOptions): Tensor {
  const extractedShape = inferShape(data);
  const shape = options?.shape ?? (extractedShape.length > 0 ? extractedShape : [1]);
  const flattened = flattenArray(data);

  const dtype = options?.dtype ?? 'float32';
  const Ctor = typedArrayCtor(dtype);
  const buffer = new Ctor(flattened) as TypedArray;

  return buildFromBuffer(shape, buffer, options);
}
