import { makeCompareOp } from './_helpers';

/**
 * Element-wise `a != b`, broadcasting. Returns a bool tensor.
 *
 * Not a standalone ONNX op — ONNX export must decompose as `Not(Equal(a, b))`.
 * We keep the composite IR node for compact graphs and let the exporter lower.
 */
export const ne = makeCompareOp('NotEqual');
