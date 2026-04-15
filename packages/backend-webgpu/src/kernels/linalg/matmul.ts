import source from './matmul.wgsl?raw';
import { WebGPUKernel, packMeta, createMetaBuffer } from '../utils';

export const matmulKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: source, label: 'MatMulShader' }),
        entryPoint: 'main',
      },
      label: 'MatMulPipeline',
    });
  },

  buildBindGroupEntries(device, _node, inputs, outputs) {
    // packMeta with no outShape: uses each tensor's own shape + strides + offset.
    // This preserves the actual strides (e.g. swapped for a transposed view).
    const metaABuf = createMetaBuffer(device, packMeta(inputs[0]));
    const metaBBuf = createMetaBuffer(device, packMeta(inputs[1]));
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

  getDispatch(_node, inputs, _outputs) {
    const shapeA = inputs[0].shape as number[];
    const shapeB = inputs[1].shape as number[];
    const M = shapeA[shapeA.length - 2] ?? 1;
    const N = shapeB[shapeB.length - 1];
    return [Math.ceil(M / 8), Math.ceil(N / 8), 1];
  },
};
