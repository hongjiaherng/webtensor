import { makeCompareOp } from './_helpers';

/** Element-wise `a > b`, broadcasting. Returns a bool tensor. ONNX: `Greater`. */
export const gt = makeCompareOp('Greater');
