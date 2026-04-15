import { WebGPUKernel } from './utils';
import { addKernel } from './elementwise/add';
import { mulKernel } from './elementwise/mul';
import { reluKernel } from './elementwise/relu';
import { matmulKernel } from './linear/matmul';
import { transposeKernel } from './shape/transpose';

export type { WebGPUKernel };

export const webgpuKernelRegistry = new Map<string, WebGPUKernel>([
  ['Add', addKernel],
  ['Mul', mulKernel],
  ['Relu', reluKernel],
  ['MatMul', matmulKernel],
  ['Transpose', transposeKernel],
]);
