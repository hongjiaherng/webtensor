import { getAddPipeline } from './elementwise/add';
import { getMulPipeline } from './elementwise/mul';
import { getReluPipeline } from './elementwise/relu';
import { getMatMulPipeline } from './linear/matmul';
import { getTransposePipeline } from './shape/transpose';

export type WebGPUPipelineFactory = (device: GPUDevice) => GPUComputePipeline;

export const webgpuKernelRegistry = new Map<string, WebGPUPipelineFactory>([
  ['Add', getAddPipeline],
  ['Mul', getMulPipeline],
  ['Relu', getReluPipeline],
  ['MatMul', getMatMulPipeline],
  ['Transpose', getTransposePipeline],
]);
