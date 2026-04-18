import source from './all.wgsl';
import {
  WebGPUKernel,
  packMeta,
  createMetaBuffer,
  createUniformBuffer,
  packReduceMeta,
  getShapeSize,
  injectMeta,
} from '../utils';

function computeReduceDims(node: { attributes?: Record<string, unknown> }, inShape: number[]) {
  const axes = ((node.attributes?.axes as number[]) ?? []).slice();
  const axisSet = new Set(axes);
  const keptAxes: number[] = [];
  for (let i = 0; i < inShape.length; i++) if (!axisSet.has(i)) keptAxes.push(i);
  const keptTotal = keptAxes.reduce((acc, a) => acc * inShape[a], 1);
  const reduceTotal = axes.reduce((acc, a) => acc * inShape[a], 1);
  return { axes, keptAxes, keptTotal, reduceTotal };
}

export const allKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: injectMeta(source), label: 'ReduceAllShader' }),
        entryPoint: 'main',
      },
      label: 'ReduceAllPipeline',
    });
  },

  buildBindGroupEntries(device, node, inputs, outputs) {
    const inShape = inputs[0].shape as number[];
    const { axes, keptAxes, keptTotal, reduceTotal } = computeReduceDims(node, inShape);

    const metaBuf = createMetaBuffer(device, packMeta(inputs[0]));
    const reduceBuf = createUniformBuffer(
      device,
      packReduceMeta(keptAxes, axes, keptTotal, reduceTotal),
    );
    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: metaBuf } },
        { binding: 3, resource: { buffer: reduceBuf } },
      ],
      tempBuffers: [metaBuf, reduceBuf],
    };
  },

  getDispatch(_node, _inputs, outputs) {
    return [Math.ceil(getShapeSize(outputs[0].shape) / 64), 1, 1];
  },
};
