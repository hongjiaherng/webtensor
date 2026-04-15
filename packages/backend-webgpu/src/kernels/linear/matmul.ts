import source from './matmul.wgsl?raw';
import { WebGPUKernel, createUniformBuffer } from '../utils';

export const matmulKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: source, label: 'MatMulShader' }), entryPoint: 'main' },
      label: 'MatMulPipeline',
    });
  },
  buildBindGroupEntries(device, _node, inputs, outputs) {
    const shapeA = inputs[0].shape as number[];
    const shapeB = inputs[1].shape as number[];
    const M = shapeA[shapeA.length - 2] || 1;
    const K = shapeA[shapeA.length - 1];
    const N = shapeB[shapeB.length - 1];
    const uniforms = createUniformBuffer(device, [M, K, N, 0]);
    return [
      { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
      { binding: 1, resource: { buffer: inputs[1].storage.buffer as GPUBuffer } },
      { binding: 2, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
      { binding: 3, resource: { buffer: uniforms } },
    ];
  },
  getDispatch(_node, inputs, outputs) {
    const shapeA = inputs[0].shape as number[];
    const shapeB = inputs[1].shape as number[];
    const M = shapeA[shapeA.length - 2] || 1;
    const N = shapeB[shapeB.length - 1];
    return [Math.ceil(M / 8), Math.ceil(N / 8), 1];
  },
};
