import type { RuntimeKernel } from '../registry';

export const squeezeView: RuntimeKernel = (node, src) => {
  const dim = node.attributes?.dim as number | undefined;
  const shape = src.shape as number[];
  const newShape: number[] = [];
  const newStrides: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    if (dim === undefined) {
      // squeeze all size-1 dimensions
      if (shape[i] !== 1) {
        newShape.push(shape[i]);
        newStrides.push(src.strides[i]);
      }
    } else {
      if (i === dim) continue; // remove this dimension (caller validated size==1)
      newShape.push(shape[i]);
      newStrides.push(src.strides[i]);
    }
  }
  return {
    storage: src.storage,
    shape: newShape,
    strides: newStrides,
    offset: src.offset,
    dtype: src.dtype,
    isView: true,
  };
};
