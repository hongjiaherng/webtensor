import source from './softmax.wgsl';
import {
  WebGPUKernel,
  packMeta,
  createMetaBuffer,
  createUniformBuffer,
  packSoftmaxMeta,
  computeContiguousStrides,
  injectMeta,
  dispatch1D,
} from '../utils';

export const softmaxKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: injectMeta(source), label: 'SoftmaxShader' }),
        entryPoint: 'main',
      },
      label: 'SoftmaxPipeline',
    });
  },

  buildBindGroupEntries(device, node, inputs, outputs) {
    const inShape = inputs[0].shape as number[];
    const axis = (node.attributes?.axis as number) ?? inShape.length - 1;
    const axisLen = inShape[axis];
    const total = inShape.reduce((acc, d) => acc * d, 1);
    const sliceCount = axisLen === 0 ? 0 : total / axisLen;
    const outStrides = computeContiguousStrides(inShape);

    const metaBuf = createMetaBuffer(device, packMeta(inputs[0]));
    const smBuf = createUniformBuffer(
      device,
      packSoftmaxMeta(axis, sliceCount, axisLen, outStrides),
    );
    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: metaBuf } },
        { binding: 3, resource: { buffer: smBuf } },
      ],
      tempBuffers: [metaBuf, smBuf],
    };
  },

  getDispatch(node, inputs, _outputs) {
    const inShape = inputs[0].shape as number[];
    const axis = (node.attributes?.axis as number) ?? inShape.length - 1;
    const axisLen = inShape[axis];
    const total = inShape.reduce((acc, d) => acc * d, 1);
    const sliceCount = axisLen === 0 ? 0 : total / axisLen;
    return dispatch1D(Math.max(sliceCount, 1));
  },
};
