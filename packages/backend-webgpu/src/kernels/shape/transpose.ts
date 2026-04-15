import source from './transpose.wgsl?raw';
import { WebGPUKernel, createUniformBuffer, flatDispatch } from '../utils';

export const transposeKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: source, label: 'TransposeShader' }), entryPoint: 'main' },
      label: 'TransposePipeline',
    });
  },
  buildBindGroupEntries(device, _node, inputs, outputs) {
    const shape = inputs[0].shape as number[];
    const M = shape[shape.length - 2] || 1;
    const N = shape[shape.length - 1];
    const uniforms = createUniformBuffer(device, [M, N, 0, 0]);
    return [
      { binding: 0, resource: { buffer: inputs[0].buffer as GPUBuffer } },
      { binding: 1, resource: { buffer: outputs[0].buffer as GPUBuffer } },
      { binding: 2, resource: { buffer: uniforms } },
    ];
  },
  getDispatch(_node, _inputs, outputs) {
    return flatDispatch(outputs);
  },
};
