import { makeCompareOp } from './_helpers';

/** Element-wise `a <= b`, broadcasting. Returns a bool tensor. ONNX: `LessOrEqual`. */
export const le = makeCompareOp('LessOrEqual');
