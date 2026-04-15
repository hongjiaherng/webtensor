// Centralized WASM bindings re-export.
// All kernel modules import wasm functions from here,
// so there's exactly one place that knows the pkg/ path.
export { default as init } from '../pkg/minitensor_wasm';
export { add, sub, mul, div, matmul, transpose } from '../pkg/minitensor_wasm';
