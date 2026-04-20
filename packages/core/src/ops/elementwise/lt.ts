import { makeCompareOp } from './_helpers';

/**
 * Element-wise `a < b`, broadcasting. Returns a bool tensor. ONNX: `Less`.
 * @category Elementwise
 */
export const lt = makeCompareOp('Less');
