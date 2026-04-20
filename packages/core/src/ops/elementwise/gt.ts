import { makeCompareOp } from './_helpers';

/**
 * Element-wise `a > b`, broadcasting. Returns a bool tensor. ONNX: `Greater`.
 * @category Elementwise
 */
export const gt = makeCompareOp('Greater');
