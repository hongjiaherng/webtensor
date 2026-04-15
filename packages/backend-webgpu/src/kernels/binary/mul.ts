import source from './mul.wgsl?raw';
import { WebGPUKernel, elementwiseBindGroupEntries, flatDispatch } from '../utils';

export const mulKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: source, label: 'MulShader' }),
        entryPoint: 'main',
      },
      label: 'MulPipeline',
    });
  },
  buildBindGroupEntries(_device, _node, inputs, outputs) {
    return elementwiseBindGroupEntries(inputs, outputs);
  },
  getDispatch(_node, _inputs, outputs) {
    return flatDispatch(outputs);
  },
};
