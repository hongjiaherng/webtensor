// Elementwise — arithmetic, unary math, comparisons, dtype cast
export { add } from './elementwise/add';
export { sub } from './elementwise/sub';
export { mul } from './elementwise/mul';
export { div } from './elementwise/div';
export { neg } from './elementwise/neg';
export { exp } from './elementwise/exp';
export { log } from './elementwise/log';
export { sqrt } from './elementwise/sqrt';
export { abs } from './elementwise/abs';
export { pow } from './elementwise/pow';
export { eq } from './elementwise/eq';
export { ne } from './elementwise/ne';
export { lt } from './elementwise/lt';
export { le } from './elementwise/le';
export { gt } from './elementwise/gt';
export { ge } from './elementwise/ge';
export { isclose } from './elementwise/isclose';
export type { IsCloseOptions } from './elementwise/isclose';
export { cast } from './elementwise/cast';

// Reduction
export { sum } from './reduction/sum';
export { mean } from './reduction/mean';
export { all } from './reduction/all';
export { any } from './reduction/any';

// Linalg
export { matmul } from './linalg/matmul';

// Activation
export { relu } from './activation/relu';
export { sigmoid } from './activation/sigmoid';
export { tanh } from './activation/tanh';
export { softmax } from './activation/softmax';

// Movement — zero-copy views + data-moving reshape ops
export { transpose } from './movement/transpose';
export { reshape } from './movement/reshape';
export { view } from './movement/view';
export { slice } from './movement/slice';
export { unsqueeze } from './movement/unsqueeze';
export { squeeze } from './movement/squeeze';
export { permute } from './movement/permute';
export { flatten } from './movement/flatten';
export { expand } from './movement/expand';
export { concat } from './movement/concat';
export { pad } from './movement/pad';

// Memory
export { contiguous } from './memory/contiguous';
export { clone } from './memory/clone';
export { detach } from './memory/detach';

// Graph inputs
export { placeholder } from './placeholder';
