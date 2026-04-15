import source from './contiguous.wgsl?raw';
import { WebGPUKernel, packMeta, createMetaBuffer, getShapeSize } from '../utils';

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

  buildBindGroupEntries(device, _node, inputs, outputs) {
    // Use the input's own shape/strides/offset — no broadcast adjustment needed.
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
