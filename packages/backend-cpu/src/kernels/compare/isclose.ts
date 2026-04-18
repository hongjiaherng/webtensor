import { CPUKernel, broadcastStridesOf, stridedIdx, buf } from '../utils';

/**
 * Element-wise `|a - b| <= atol + rtol * |b|`. Float32-only. Tolerances arrive
 * as node attributes from the core op. NaN/NaN is equal only when
 * `equalNan === 1`. ±inf compares equal to itself.
 */
export const iscloseKernel: CPUKernel = (node, inputs, outputs) => {
  const rtol = node.attributes?.rtol as number;
  const atol = node.attributes?.atol as number;
  const equalNan = (node.attributes?.equalNan as number) === 1;

  const aShape = inputs[0].shape as number[];
  const bShape = inputs[1].shape as number[];
  const outShape = outputs[0].shape as number[];
  const aBuf = buf(inputs[0]);
  const bBuf = buf(inputs[1]);
  const outBuf = buf(outputs[0]);
  const total = outBuf.length;

  const aBcast = broadcastStridesOf(outShape, aShape, inputs[0].strides);
  const bBcast = broadcastStridesOf(outShape, bShape, inputs[1].strides);

  for (let i = 0; i < total; i++) {
    const av = aBuf[stridedIdx(outShape, aBcast, inputs[0].offset, i)];
    const bv = bBuf[stridedIdx(outShape, bBcast, inputs[1].offset, i)];

    let close: boolean;
    if (Number.isNaN(av) || Number.isNaN(bv)) {
      close = equalNan && Number.isNaN(av) && Number.isNaN(bv);
    } else if (!Number.isFinite(av) || !Number.isFinite(bv)) {
      // Infinities are close only when they match exactly (+inf vs +inf).
      // Without this, `|inf - (-inf)| <= atol + rtol*inf` evaluates to true.
      close = av === bv;
    } else {
      close = Math.abs(av - bv) <= atol + rtol * Math.abs(bv);
    }
    outBuf[i] = close ? 1 : 0;
  }
};
