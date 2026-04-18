import { makeCompareOp } from './_factory';

/** Element-wise `a >= b`, broadcasting. Returns a bool tensor. ONNX: `GreaterOrEqual`. */
export const ge = makeCompareOp('GreaterOrEqual');
