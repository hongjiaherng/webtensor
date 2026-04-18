import { makeCompareOp } from './_factory';

/** Element-wise `a < b`, broadcasting. Returns a bool tensor. ONNX: `Less`. */
export const lt = makeCompareOp('Less');
