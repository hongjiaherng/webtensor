import { CPUKernel } from './utils';
import { addKernel } from './binary/add';
import { subKernel } from './binary/sub';
import { mulKernel } from './binary/mul';
import { divKernel } from './binary/div';
import { reluKernel, reluGradKernel } from './unary/relu';
import { matmulKernel } from './linalg/matmul';
import { contiguousKernel } from './memory/contiguous';

export type { CPUKernel };

export const cpuKernelRegistry = new Map<string, CPUKernel>([
  ['Add', addKernel],
  ['Sub', subKernel],
  ['Mul', mulKernel],
  ['Div', divKernel],
  ['Relu', reluKernel],
  ['ReluGrad', reluGradKernel],
  ['MatMul', matmulKernel],
  ['Contiguous', contiguousKernel],
]);
