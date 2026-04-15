import { CPUKernel } from './utils';
import { addKernel } from './elementwise/add';
import { subKernel } from './elementwise/sub';
import { mulKernel } from './elementwise/mul';
import { divKernel } from './elementwise/div';
import { reluKernel, reluGradKernel } from './elementwise/relu';
import { matmulKernel } from './linear/matmul';
import { transposeKernel } from './shape/transpose';

export type { CPUKernel };

export const cpuKernelRegistry = new Map<string, CPUKernel>([
  ['Add', addKernel],
  ['Sub', subKernel],
  ['Mul', mulKernel],
  ['Div', divKernel],
  ['Relu', reluKernel],
  ['ReluGrad', reluGradKernel],
  ['MatMul', matmulKernel],
  ['Transpose', transposeKernel],
]);
