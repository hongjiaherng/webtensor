import source from './backward.wgsl';
import { WebGPUKernel, packMeta, createMetaBuffer, getShapeSize, injectMeta } from '../../utils';

export const reluBackwardKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: injectMeta(source),
          label: 'ReluBackwardShader',
        }),
        entryPoint: 'main',
      },
      label: 'ReluBackwardPipeline',
    });
  },

  buildBindGroupEntries(device, _node, inputs, outputs) {
    const metaGrad = createMetaBuffer(device, packMeta(inputs[0]));
    const metaA = createMetaBuffer(device, packMeta(inputs[1]));
    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: inputs[1].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 3, resource: { buffer: metaGrad } },
        { binding: 4, resource: { buffer: metaA } },
      ],
      tempBuffers: [metaGrad, metaA],
    };
  },

  getDispatch(_node, _inputs, outputs) {
    return [Math.ceil(getShapeSize(outputs[0].shape) / 64), 1, 1];
  },
};
