import { WASMKernel } from './utils';
import { addKernel } from './binary/add';
import { subKernel } from './binary/sub';
import { mulKernel } from './binary/mul';
import { divKernel } from './binary/div';
import { reluKernel } from './activation/relu';
import { reluGradKernel } from './activation/reluGrad';
import { negKernel } from './unary/neg';
import { expKernel } from './unary/exp';
import { logKernel } from './unary/log';
import { sqrtKernel } from './unary/sqrt';
import { absKernel } from './unary/abs';
import { powKernel } from './unary/pow';
import { sigmoidKernel } from './activation/sigmoid';
import { tanhKernel } from './activation/tanh';
import { matmulKernel } from './linalg/matmul';
import { reduceSumKernel } from './reduce/reduceSum';
import { reduceMeanKernel } from './reduce/reduceMean';
import { softmaxKernel } from './activation/softmax';
import { contiguousKernel } from './memory/contiguous';
import { castKernel } from './cast/cast';
import { eqKernel } from './compare/eq';
import { neKernel } from './compare/ne';
import { ltKernel } from './compare/lt';
import { leKernel } from './compare/le';
import { gtKernel } from './compare/gt';
import { geKernel } from './compare/ge';
import { iscloseKernel } from './compare/isclose';
import { concatKernel } from './join/concat';
import { padKernel } from './padding/pad';

export type { WASMKernel };

export const wasmKernelRegistry = new Map<string, WASMKernel>([
  ['Add', addKernel],
  ['Sub', subKernel],
  ['Mul', mulKernel],
  ['Div', divKernel],
  ['Relu', reluKernel],
  ['ReluGrad', reluGradKernel],
  ['Neg', negKernel],
  ['Exp', expKernel],
  ['Log', logKernel],
  ['Sqrt', sqrtKernel],
  ['Abs', absKernel],
  ['Pow', powKernel],
  ['Sigmoid', sigmoidKernel],
  ['Tanh', tanhKernel],
  ['MatMul', matmulKernel],
  ['ReduceSum', reduceSumKernel],
  ['ReduceMean', reduceMeanKernel],
  ['Softmax', softmaxKernel],
  ['Contiguous', contiguousKernel],
  ['Cast', castKernel],
  ['Equal', eqKernel],
  ['NotEqual', neKernel],
  ['Less', ltKernel],
  ['LessOrEqual', leKernel],
  ['Greater', gtKernel],
  ['GreaterOrEqual', geKernel],
  ['IsClose', iscloseKernel],
  ['Concat', concatKernel],
  ['Pad', padKernel],
]);
