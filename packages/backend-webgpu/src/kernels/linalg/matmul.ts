import source from './matmul.wgsl';
import {
  WebGPUKernel,
  createMetaBuffer,
  createUniformBuffer,
  packMetaMatMulInput,
  packBatchMeta,
  injectMeta,
} from '../utils';

function matmulDims(
  inputs: { shape: (number | null)[] }[],
  outputs: { shape: (number | null)[] }[],
) {
  const aShape = inputs[0].shape as number[];
  const bShape = inputs[1].shape as number[];
  const outShape = outputs[0].shape as number[];
  const rankA = aShape.length;
  const rankB = bShape.length;
  const outRank = outShape.length;
  const batchOutShape = outShape.slice(0, outRank - 2);
  const M = aShape[rankA - 2];
  const K = aShape[rankA - 1];
  const N = bShape[rankB - 1];
  const batchTotal = batchOutShape.reduce((acc, d) => acc * d, 1);
  return { batchOutShape, M, K, N, batchTotal };
}

export const matmulKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: injectMeta(source), label: 'MatMulShader' }),
        entryPoint: 'main',
      },
      label: 'MatMulPipeline',
    });
  },

  buildBindGroupEntries(device, _node, inputs, outputs) {
    const { batchOutShape, M, K, N } = matmulDims(inputs, outputs);
    const metaA = packMetaMatMulInput(inputs[0], batchOutShape);
    const metaB = packMetaMatMulInput(inputs[1], batchOutShape);
    const batch = packBatchMeta(batchOutShape, M, K, N);

    const metaABuf = createMetaBuffer(device, metaA);
    const metaBBuf = createMetaBuffer(device, metaB);
    const batchBuf = createUniformBuffer(device, batch);
    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: inputs[1].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 3, resource: { buffer: metaABuf } },
        { binding: 4, resource: { buffer: metaBBuf } },
        { binding: 5, resource: { buffer: batchBuf } },
      ],
      tempBuffers: [metaABuf, metaBBuf, batchBuf],
    };
  },

  getDispatch(_node, inputs, outputs) {
    const { M, N, batchTotal } = matmulDims(inputs, outputs);
    return [Math.ceil(M / 8), Math.ceil(N / 8), Math.max(batchTotal, 1)];
  },
};
