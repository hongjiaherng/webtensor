import { WASMKernel } from './utils';
import { addKernel } from './binary/add';
import { subKernel } from './binary/sub';
import { mulKernel } from './binary/mul';
import { divKernel } from './binary/div';
import { reluKernel, reluGradKernel } from './unary/relu';
import { negKernel } from './unary/neg';
import { expKernel } from './unary/exp';
import { logKernel } from './unary/log';
import { sqrtKernel } from './unary/sqrt';
import { absKernel } from './unary/abs';
import { powKernel } from './unary/pow';
import { sigmoidKernel } from './unary/sigmoid';
import { tanhKernel } from './unary/tanh';
import { matmulKernel } from './linalg/matmul';
import { reduceSumKernel } from './reduce/reduceSum';
import { reduceMeanKernel } from './reduce/reduceMean';
import { softmaxKernel } from './activation/softmax';
import { contiguousKernel } from './memory/contiguous';

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
]);
