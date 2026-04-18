export const VERSION = '0.0.0';

// ---------------------------------------------------------------------------
// Public API — what the typical user imports.

// Types
export type { DType } from '@webtensor/ir';
export type { Device, NestedArray } from './types';
export type { InitOptions } from './init';
export type { CompileOptions, ShapeLike, FeedValue } from './compile';
export type { RunOptions } from './run';

// The Tensor class
export { Tensor } from './tensor';

// Factory functions
export { tensor } from './init/tensor';
export { zeros } from './init/zeros';
export { ones } from './init/ones';
export { rand } from './init/rand';
export { randn } from './init/randn';
export { zerosLike, onesLike, randnLike } from './init/like';

// Ops — math primitives and views. (Activations live in @webtensor/nn.)
export * from './ops';

// Compile / grad / run — the high-level training API
export { compile, grad } from './compile';
export { run } from './run';

// Comparison (PyTorch / JAX parity: torch.equal, torch.allclose)
export { equal, allclose } from './compare';
export type { AllcloseOptions } from './compare';

// ---------------------------------------------------------------------------
// Advanced / low-level — exported for custom compilers, ONNX integration,
// and library extensions. Most users do not need these.

export { compileGraph } from './compile';
export { Engine, registerBackend } from '@webtensor/runtime';
export type { Backend, RuntimeTensor, RuntimeStorage, TypedArray } from '@webtensor/runtime';
export type { Node, Value, Graph, AttributeValue } from '@webtensor/ir';
