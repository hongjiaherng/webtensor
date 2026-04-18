import { makeCompareOp } from './_helpers';

/** Element-wise `a < b`, broadcasting. Returns a bool tensor. ONNX: `Less`. */
export const lt = makeCompareOp('Less');
