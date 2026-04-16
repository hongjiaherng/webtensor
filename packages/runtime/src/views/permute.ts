import { ViewFn } from './types';

export const permuteView: ViewFn = (node, src) => {
  const axes = node.attributes!.axes as number[];
  const newShape = axes.map(i => src.shape[i]);
  const newStrides = axes.map(i => src.strides[i]);
  return {
    storage: src.storage,
    shape: newShape,
    strides: newStrides,
    offset: src.offset,
    dtype: src.dtype,
    isView: true,
  };
};
