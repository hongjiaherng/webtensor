import { CPUKernel, stridedIdx, buf } from '../utils';

/**
 * Dtype conversion. Reads the input with the input dtype's buffer view, writes
 * the output with the output dtype's buffer view. Coercion rules match JS
 * typed-array store semantics (float→int truncates toward zero). Bool outputs
 * use Uint8Array where 0/1 encode false/true.
 */
export const castKernel: CPUKernel = (_node, inputs, outputs) => {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  const offset = inputs[0].offset;
  const inBuf = buf(inputs[0]);
  const outBuf = buf(outputs[0]);
  const outDType = outputs[0].dtype;

  if (outDType === 'bool') {
    // Any non-zero → 1, zero → 0. Uint8Array stores already do the (v & 0xff)
    // coercion but we want the boolean canonicalization explicit.
    for (let i = 0; i < outBuf.length; i++) {
      const v = inBuf[stridedIdx(shape, strides, offset, i)];
      outBuf[i] = v !== 0 ? 1 : 0;
    }
  } else {
    // float32 ↔ int32 and bool → float32/int32: straight store, rely on the
    // typed array's native coercion (Int32Array truncates float→int toward 0,
    // Float32Array widens int→float exactly).
    for (let i = 0; i < outBuf.length; i++) {
      outBuf[i] = inBuf[stridedIdx(shape, strides, offset, i)];
    }
  }
};
