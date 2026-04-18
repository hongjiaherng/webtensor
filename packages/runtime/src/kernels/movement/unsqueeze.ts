import type { RuntimeKernel } from '../registry';

export const unsqueezeView: RuntimeKernel = (node, src) => {
  const dim = node.attributes!.dim as number;
  const newShape = [...(src.shape as number[])];
  const newStrides = [...src.strides];
  // Insert size-1 dimension. The stride for a size-1 dim doesn't matter
  // for indexing, but we use the next inner stride for consistency.
  const insertStride = dim < newStrides.length ? newStrides[dim] * (newShape[dim] as number) : 1;
  newShape.splice(dim, 0, 1);
  newStrides.splice(dim, 0, insertStride);
  return {
    storage: src.storage,
    shape: newShape,
    strides: newStrides,
    offset: src.offset,
    dtype: src.dtype,
    isView: true,
  };
};
