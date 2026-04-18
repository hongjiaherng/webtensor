import { CPUKernel, broadcastStridesOf, stridedIdx, buf } from '../utils';

/**
 * Element-wise `isclose`. Float32-only. Formula:
 *   `|a - b| <= atol + rtol * |b|`
 *
 * Matches NumPy / PyTorch / JAX semantics — NaN values only compare equal
 * when `equalNan === true`; ±∞ only compares equal to the same signed infinity.
 * WASM (Rust) and WebGPU (WGSL) re-implement the same predicate — keep them
 * in sync with this kernel.
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

    const aNan = Number.isNaN(av);
    const bNan = Number.isNaN(bv);
    let ok: boolean;
    if (aNan || bNan) {
      ok = aNan && bNan && equalNan;
    } else if (!Number.isFinite(av) || !Number.isFinite(bv)) {
      ok = av === bv;
    } else {
      ok = Math.abs(av - bv) <= atol + rtol * Math.abs(bv);
    }
    outBuf[i] = ok ? 1 : 0;
  }
};
