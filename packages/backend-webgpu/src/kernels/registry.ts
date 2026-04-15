import { WebGPUKernel } from './utils';
import { addKernel } from './binary/add';
import { subKernel } from './binary/sub';
import { mulKernel } from './binary/mul';
import { divKernel } from './binary/div';
import { reluKernel } from './unary/relu';
import { matmulKernel } from './linalg/matmul';
import { contiguousKernel } from './memory/contiguous';

export type { WebGPUKernel };

export const webgpuKernelRegistry = new Map<string, WebGPUKernel>([
  ['Add', addKernel],
  ['Sub', subKernel],
  ['Mul', mulKernel],
  ['Div', divKernel],
  ['Relu', reluKernel],
  ['MatMul', matmulKernel],
  ['Contiguous', contiguousKernel],
]);
