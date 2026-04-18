import { makeCompareOp } from './_factory';

/** Element-wise `a > b`, broadcasting. Returns a bool tensor. ONNX: `Greater`. */
export const gt = makeCompareOp('Greater');
