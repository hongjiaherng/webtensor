import source from './tanh.wgsl';
import { WebGPUKernel, packMeta, createMetaBuffer, getShapeSize, injectMeta } from '../utils';

export const tanhKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: injectMeta(source), label: 'TanhShader' }),
        entryPoint: 'main',
      },
      label: 'TanhPipeline',
    });
  },

  buildBindGroupEntries(device, _node, inputs, outputs) {
    const metaBuf = createMetaBuffer(device, packMeta(inputs[0]));
    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: metaBuf } },
      ],
      tempBuffers: [metaBuf],
    };
  },

  getDispatch(_node, _inputs, outputs) {
    return [Math.ceil(getShapeSize(outputs[0].shape) / 64), 1, 1];
  },
};
