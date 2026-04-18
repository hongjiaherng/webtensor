import { makeCompareOp } from './_helpers';

/** Element-wise `a == b`, broadcasting. Returns a bool tensor. ONNX: `Equal`. */
export const eq = makeCompareOp('Equal');
