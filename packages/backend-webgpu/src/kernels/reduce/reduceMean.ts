import source from './reduceMean.wgsl';
import {
  WebGPUKernel,
  packMeta,
  createMetaBuffer,
  createUniformBuffer,
  packReduceMeta,
  getShapeSize,
} from '../utils';

export const reduceMeanKernel: WebGPUKernel = {
  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: source, label: 'ReduceMeanShader' }),
        entryPoint: 'main',
      },
      label: 'ReduceMeanPipeline',
    });
  },

  buildBindGroupEntries(device, node, inputs, outputs) {
    const inShape = inputs[0].shape as number[];
    const axes = ((node.attributes?.axes as number[]) ?? []).slice();
    const axisSet = new Set(axes);
    const keptAxes: number[] = [];
    for (let i = 0; i < inShape.length; i++) if (!axisSet.has(i)) keptAxes.push(i);
    const keptTotal = keptAxes.reduce((acc, a) => acc * inShape[a], 1);
    const reduceTotal = axes.reduce((acc, a) => acc * inShape[a], 1);

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
