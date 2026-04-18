// Binary
export { add } from './binary/add';
export { sub } from './binary/sub';
export { mul } from './binary/mul';
export { div } from './binary/div';

// Unary math primitives (NumPy parity — ML-flavored activations live in @webtensor/nn)
export { neg } from './unary/neg';
export { exp } from './unary/exp';
export { log } from './unary/log';
export { sqrt } from './unary/sqrt';
export { abs } from './unary/abs';
export { pow } from './unary/pow';

// Linalg
export { matmul } from './linalg/matmul';

// Reductions
export { sum, reduceSum } from './reduce/sum';
export { mean, reduceMean } from './reduce/mean';

// Views (zero-copy)
export { transpose } from './view/transpose';
export { reshape } from './view/reshape';
export { view } from './view/view';
export { slice } from './view/slice';
export { unsqueeze } from './view/unsqueeze';
export { squeeze } from './view/squeeze';
export { permute } from './view/permute';
export { flatten } from './view/flatten';
export { expand } from './view/expand';

// Memory
export { contiguous } from './memory/contiguous';
export { clone } from './memory/clone';
export { detach } from './memory/detach';

// Graph inputs
export { placeholder } from './placeholder';
