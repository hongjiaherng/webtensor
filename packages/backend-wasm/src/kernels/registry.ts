import { WASMKernel } from './utils';
import { addKernel } from './elementwise/add';
import { subKernel } from './elementwise/sub';
import { mulKernel } from './elementwise/mul';
import { divKernel } from './elementwise/div';
import { negKernel } from './elementwise/neg';
import { expKernel } from './elementwise/exp';
import { logKernel } from './elementwise/log';
import { sqrtKernel } from './elementwise/sqrt';
import { absKernel } from './elementwise/abs';
import { powKernel } from './elementwise/pow';
import { eqKernel } from './elementwise/eq';
import { neKernel } from './elementwise/ne';
import { ltKernel } from './elementwise/lt';
import { leKernel } from './elementwise/le';
import { gtKernel } from './elementwise/gt';
import { geKernel } from './elementwise/ge';
import { iscloseKernel } from './elementwise/isclose';
import { castKernel } from './elementwise/cast';
import { reduceSumKernel } from './reduction/sum';
import { reduceMeanKernel } from './reduction/mean';
import { allKernel } from './reduction/all';
import { anyKernel } from './reduction/any';
import { matmulKernel } from './linalg/matmul';
import { reluKernel } from './activation/relu/forward';
import { reluBackwardKernel } from './activation/relu/backward';
import { sigmoidKernel } from './activation/sigmoid';
import { tanhKernel } from './activation/tanh';
import { softmaxKernel } from './activation/softmax';
import { contiguousKernel } from './memory/contiguous';
import { concatKernel } from './movement/concat';
import { padKernel } from './movement/pad';

export type { WASMKernel };

export const wasmKernelRegistry = new Map<string, WASMKernel>([
  ['Add', addKernel],
  ['Sub', subKernel],
  ['Mul', mulKernel],
  ['Div', divKernel],
  ['Neg', negKernel],
  ['Exp', expKernel],
  ['Log', logKernel],
  ['Sqrt', sqrtKernel],
  ['Abs', absKernel],
  ['Pow', powKernel],
  ['Equal', eqKernel],
  ['NotEqual', neKernel],
  ['Less', ltKernel],
  ['LessOrEqual', leKernel],
  ['Greater', gtKernel],
  ['GreaterOrEqual', geKernel],
  ['IsClose', iscloseKernel],
  ['Cast', castKernel],
  ['ReduceSum', reduceSumKernel],
  ['ReduceMean', reduceMeanKernel],
  ['ReduceAll', allKernel],
  ['ReduceAny', anyKernel],
  ['MatMul', matmulKernel],
  ['Relu', reluKernel],
  ['ReluBackward', reluBackwardKernel],
  ['Sigmoid', sigmoidKernel],
  ['Tanh', tanhKernel],
  ['Softmax', softmaxKernel],
  ['Contiguous', contiguousKernel],
  ['Concat', concatKernel],
  ['Pad', padKernel],
]);
