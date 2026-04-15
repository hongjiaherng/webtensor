import { stridedIdx } from '@minitensor/runtime';
import { WASMKernel, handleOf, getShapeSize } from '../utils';
import { getF32View } from '../../module';

export const contiguousKernel: WASMKernel = (module, _node, inputs, outputs) => {
  const src = inputs[0];
  const shape = src.shape as number[];
  const total = getShapeSize(shape);
  const srcView = getF32View(module, handleOf(inputs[0]));
  const dstView = getF32View(module, handleOf(outputs[0]));
  for (let i = 0; i < total; i++) {
    dstView[i] = srcView[stridedIdx(shape, src.strides, src.offset, i)];
  }
};
