import { CPUKernel, buf } from '../utils';

export const softmaxKernel: CPUKernel = (node, inputs, outputs) => {
  const inShape = inputs[0].shape as number[];
  const inStrides = inputs[0].strides;
  const inOffset = inputs[0].offset;
  const inBuf = buf(inputs[0]);
  const outBuf = buf(outputs[0]);

  const axis = node.attributes!.axis as number;
  const rank = inShape.length;
  const axisLen = inShape[axis];

  // Per-slice: enumerate "outer" positions (all non-axis coords)
  const otherAxes: number[] = [];
  for (let i = 0; i < rank; i++) if (i !== axis) otherAxes.push(i);
  const otherShape = otherAxes.map((a) => inShape[a]);
  const otherTotal = otherShape.reduce((acc, d) => acc * d, 1);

  const coord = new Array<number>(rank).fill(0);

  const outStrides = outputs[0].strides;
  const outAxisStride = outStrides[axis];

  for (let o = 0; o < otherTotal; o++) {
    let rem = o;
    for (let i = otherAxes.length - 1; i >= 0; i--) {
      const a = otherAxes[i];
      const d = inShape[a];
      coord[a] = rem % d;
      rem = Math.floor(rem / d);
    }

    // Base input offset for this slice (axis coord = 0)
    coord[axis] = 0;
    let inBase = inOffset;
    for (let d = 0; d < rank; d++) inBase += coord[d] * inStrides[d];
    const inAxisStride = inStrides[axis];

    // Base output offset for this slice (out is contiguous)
    let outBase = 0;
    for (let d = 0; d < rank; d++) outBase += coord[d] * outStrides[d];

    // Pass 1: max
    let maxV = -Infinity;
    for (let k = 0; k < axisLen; k++) {
      const v = inBuf[inBase + k * inAxisStride];
      if (v > maxV) maxV = v;
    }

    // Pass 2: exp(x - max), sum
    let sum = 0;
    for (let k = 0; k < axisLen; k++) {
      const e = Math.exp(inBuf[inBase + k * inAxisStride] - maxV);
      outBuf[outBase + k * outAxisStride] = e;
      sum += e;
    }

    // Pass 3: divide
    const invSum = sum === 0 ? 0 : 1 / sum;
    for (let k = 0; k < axisLen; k++) {
      outBuf[outBase + k * outAxisStride] *= invSum;
    }
  }
};
