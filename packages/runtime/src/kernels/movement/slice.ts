import type { RuntimeKernel } from '../registry';

export const sliceView: RuntimeKernel = (node, src) => {
  const starts = node.attributes!.starts as number[];
  const ends = node.attributes!.ends as number[];
  const newShape = starts.map((s, i) => ends[i] - s);
  const newOffset = src.offset + starts.reduce((acc, s, i) => acc + s * src.strides[i], 0);
  return {
    storage: src.storage,
    shape: newShape,
    strides: [...src.strides],
    offset: newOffset,
    dtype: src.dtype,
    isView: true,
  };
};
