import type { RuntimeKernel } from '../registry';

export const transposeView: RuntimeKernel = (_node, src) => {
  const rank = src.shape.length;
  const newShape = [...src.shape];
  const newStrides = [...src.strides];
  newShape[rank - 1] = src.shape[rank - 2];
  newShape[rank - 2] = src.shape[rank - 1];
  newStrides[rank - 1] = src.strides[rank - 2];
  newStrides[rank - 2] = src.strides[rank - 1];
  return {
    storage: src.storage,
    shape: newShape,
    strides: newStrides,
    offset: src.offset,
    dtype: src.dtype,
    isView: true,
  };
};
