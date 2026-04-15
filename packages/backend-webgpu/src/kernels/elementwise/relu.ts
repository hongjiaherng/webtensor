import source from './relu.wgsl?raw';
import { WebGPUKernel, elementwiseBindGroupEntries, flatDispatch } from '../utils';

export const reluKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: source, label: 'ReluShader' }), entryPoint: 'main' },
      label: 'ReluPipeline',
    });
  },
  buildBindGroupEntries(_device, _node, inputs, outputs) {
    return elementwiseBindGroupEntries(inputs, outputs);
  },
  getDispatch(_node, _inputs, outputs) {
    return flatDispatch(outputs);
  },
};
