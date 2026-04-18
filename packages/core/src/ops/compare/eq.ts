import { makeCompareOp } from './_factory';

/** Element-wise `a == b`, broadcasting. Returns a bool tensor. ONNX: `Equal`. */
export const eq = makeCompareOp('Equal');
