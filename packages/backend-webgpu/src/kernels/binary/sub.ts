import source from './sub.wgsl?raw';
import { WebGPUKernel, packMeta, createMetaBuffer, getShapeSize } from '../utils';

export const subKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: source, label: 'SubShader' }),
        entryPoint: 'main',
      },
      label: 'SubPipeline',
    });
  },

  buildBindGroupEntries(device, _node, inputs, outputs) {
    const outShape = outputs[0].shape as number[];
    const metaABuf = createMetaBuffer(device, packMeta(inputs[0], outShape));
    const metaBBuf = createMetaBuffer(device, packMeta(inputs[1], outShape));
    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: inputs[1].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 3, resource: { buffer: metaABuf } },
        { binding: 4, resource: { buffer: metaBBuf } },
      ],
      tempBuffers: [metaABuf, metaBBuf],
    };
  },

  getDispatch(_node, _inputs, outputs) {
    return [Math.ceil(getShapeSize(outputs[0].shape) / 64), 1, 1];
  },
};
