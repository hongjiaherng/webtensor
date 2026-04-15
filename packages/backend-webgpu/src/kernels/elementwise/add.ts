import source from './add.wgsl?raw';
import { WebGPUKernel, elementwiseBindGroupEntries, flatDispatch } from '../utils';

export const addKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: source, label: 'AddShader' }), entryPoint: 'main' },
      label: 'AddPipeline',
    });
  },
  buildBindGroupEntries(_device, _node, inputs, outputs) {
    return elementwiseBindGroupEntries(inputs, outputs);
  },
  getDispatch(_node, _inputs, outputs) {
    return flatDispatch(outputs);
  },
};
