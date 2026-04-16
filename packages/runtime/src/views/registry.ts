import { ViewFn } from './types';
import { transposeView } from './transpose';
import { sliceView } from './slice';
import { unsqueezeView } from './unsqueeze';
import { squeezeView } from './squeeze';
import { permuteView } from './permute';
import { expandView } from './expand';

export type { ViewFn };

export const viewRegistry = new Map<string, ViewFn>([
  ['Transpose', transposeView],
  ['Slice', sliceView],
  ['Unsqueeze', unsqueezeView],
  ['Squeeze', squeezeView],
  ['Permute', permuteView],
  ['Expand', expandView],
]);
