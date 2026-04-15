import source from './contiguous.wgsl?raw';
import { WebGPUKernel, getShapeSize } from '../utils';

export const contiguousKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: source, label: 'ContiguousShader' }),
        entryPoint: 'main',
      },
      label: 'ContiguousPipeline',
    });
  },

  buildBindGroupEntries(_device, _node, inputs, outputs) {
    return [
      { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
      { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
    ];
  },

  getDispatch(_node, _inputs, outputs) {
    return [Math.ceil(getShapeSize(outputs[0].shape) / 64), 1, 1];
  },
};
