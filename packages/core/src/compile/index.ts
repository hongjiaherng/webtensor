// Low-level: trace an eager Tensor graph into an IR Graph. Used by `compile()`
// and `run()` internally; also exported for advanced users (ONNX export, etc.).
export { compileGraph } from './trace';

// High-level: JAX-style `@jit` compile.
export { compile } from './jit';
export type { CompileOptions, ShapeLike, FeedValue } from './jit';
