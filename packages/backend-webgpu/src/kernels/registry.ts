import { WebGPUKernel } from './utils';
import { addKernel } from './binary/add';
import { mulKernel } from './binary/mul';
import { reluKernel } from './unary/relu';
import { matmulKernel } from './linalg/matmul';
import { contiguousKernel } from './memory/contiguous';

export type { WebGPUKernel };

export const webgpuKernelRegistry = new Map<string, WebGPUKernel>([
  ['Add', addKernel],
  ['Mul', mulKernel],
  ['Relu', reluKernel],
  ['MatMul', matmulKernel],
  ['Contiguous', contiguousKernel],
]);
