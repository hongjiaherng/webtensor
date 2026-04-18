import { transposeView } from './movement/transpose';
import { sliceView } from './movement/slice';
import { unsqueezeView } from './movement/unsqueeze';
import { squeezeView } from './movement/squeeze';
import { permuteView } from './movement/permute';
import { expandView } from './movement/expand';

import { Node } from '@webtensor/ir';
import { RuntimeTensor } from '../backend';

/**
 * A runtime kernel — a "no-op" view kernel that produces a new `RuntimeTensor`
 * by recomputing `shape` / `strides` / `offset` metadata only. No storage is
 * allocated and no backend (CPU / WASM / WebGPU) is invoked: the output tensor
 * shares the input's backing buffer.
 *
 * Contrast with backend kernels (`CPUKernel`, `WASMKernel`, `WebGPUKernel`),
 * which dispatch real device compute. Runtime kernels are intercepted by the
 * `Engine` before backend dispatch — they are free at execution time.
 */
export type RuntimeKernel = (node: Node, src: RuntimeTensor) => RuntimeTensor;

/**
 * Registry of zero-copy view kernels handled host-side by the `Engine`.
 *
 * When the `Engine` encounters a node whose op is in this registry, it skips
 * backend dispatch entirely and runs the registered function to produce a new
 * `RuntimeTensor` that shares storage with its input — a pure metadata
 * transform (reshape of strides/offset/shape). This is why "view ops" don't
 * appear in any backend kernel registry: they are no-op kernels, resolved by
 * the runtime itself.
 *
 * Note: `Reshape` and `View` are *not* in this registry even though they are
 * view-like — they need special handling in `Engine` because `Reshape` may
 * auto-copy on non-contiguous input (PyTorch semantics) and `View` must
 * verify contiguity.
 */
export const runtimeKernelRegistry = new Map<string, RuntimeKernel>([
  ['Transpose', transposeView],
  ['Slice', sliceView],
  ['Unsqueeze', unsqueezeView],
  ['Squeeze', squeezeView],
  ['Permute', permuteView],
  ['Expand', expandView],
]);
