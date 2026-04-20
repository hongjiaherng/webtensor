import { makeCompareOp } from './_helpers';

/**
 * Element-wise `a >= b`, broadcasting. Returns a bool tensor. ONNX: `GreaterOrEqual`.
 * @category Elementwise
 */
export const ge = makeCompareOp('GreaterOrEqual');
