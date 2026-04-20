import source from './pow.wgsl';
import {
  WebGPUKernel,
  packMeta,
  createMetaBuffer,
  getShapeSize,
  injectMeta,
  dispatch1D,
} from '../utils';

export const powKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: injectMeta(source), label: 'PowShader' }),
        entryPoint: 'main',
      },
      label: 'PowPipeline',
    });
  },

  buildBindGroupEntries(device, node, inputs, outputs) {
    const metaBuf = createMetaBuffer(device, packMeta(inputs[0]));
    const exponent = node.attributes!.exponent as number;
    const expData = new Float32Array([exponent]);
    const expBuf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(expBuf, 0, expData);
    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: metaBuf } },
        { binding: 3, resource: { buffer: expBuf } },
      ],
      tempBuffers: [metaBuf, expBuf],
    };
  },

  getDispatch(_node, _inputs, outputs) {
    return dispatch1D(getShapeSize(outputs[0].shape));
  },
};
