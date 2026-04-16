import { ViewFn } from './types';

export const expandView: ViewFn = (node, src) => {
  const targetShape = node.attributes!.shape as number[];
  const srcShape = src.shape as number[];
  const srcStrides = src.strides;
  const outRank = targetShape.length;
  const rankOffset = outRank - srcShape.length;
  const newStrides = new Array<number>(outRank).fill(0);
  for (let i = 0; i < srcShape.length; i++) {
    // Dimensions that are not broadcast keep their stride; broadcast dims get stride 0.
    newStrides[rankOffset + i] = srcShape[i] === 1 ? 0 : srcStrides[i];
  }
  return {
    storage: src.storage,
    shape: targetShape,
    strides: newStrides,
    offset: src.offset,
    dtype: src.dtype,
    isView: true,
  };
};
