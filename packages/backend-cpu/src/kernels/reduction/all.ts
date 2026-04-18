import { CPUKernel, buf } from '../utils';

export const allKernel: CPUKernel = (node, inputs, outputs) => {
  const inShape = inputs[0].shape as number[];
  const inStrides = inputs[0].strides;
  const inOffset = inputs[0].offset;
  const inBuf = buf(inputs[0]);
  const outBuf = buf(outputs[0]);

  const axes = (node.attributes!.axes as number[]) ?? [];
  const inRank = inShape.length;
  const axisSet = new Set(axes);

  const keptAxes: number[] = [];
  for (let i = 0; i < inRank; i++) if (!axisSet.has(i)) keptAxes.push(i);

  const keptTotal = keptAxes.reduce((acc, a) => acc * inShape[a], 1);
  const reduceShape = axes.map((a) => inShape[a]);
  const reduceTotal = reduceShape.reduce((acc, d) => acc * d, 1);

  const inCoord = new Array<number>(inRank).fill(0);

  for (let outIdx = 0; outIdx < keptTotal; outIdx++) {
    let rem = outIdx;
    for (let i = keptAxes.length - 1; i >= 0; i--) {
      const a = keptAxes[i];
      const d = inShape[a];
      inCoord[a] = rem % d;
      rem = Math.floor(rem / d);
    }

    let acc = 1;
    for (let rIdx = 0; rIdx < reduceTotal; rIdx++) {
      let rRem = rIdx;
      for (let i = axes.length - 1; i >= 0; i--) {
        const a = axes[i];
        const d = inShape[a];
        inCoord[a] = rRem % d;
        rRem = Math.floor(rRem / d);
      }
      let off = inOffset;
      for (let d = 0; d < inRank; d++) off += inCoord[d] * inStrides[d];
      if (inBuf[off] === 0) {
        acc = 0;
        break;
      }
    }

    outBuf[outIdx] = acc;
  }
};
