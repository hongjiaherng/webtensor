import source from './concat.wgsl';
import {
  WebGPUKernel,
  packMeta,
  createMetaBuffer,
  createUniformBuffer,
  getShapeSize,
  renderWgsl,
} from '../utils';

/**
 * Concat — one pipeline, N dispatches (one per input). Uses the
 * `executeOverride` hook to encode all dispatches onto the shared encoder.
 * Each dispatch scatters one input into the output along `axis`, starting at
 * the running `axisStart`.
 */
export const concatKernel: WebGPUKernel = {
  pipelineKey(_node, _inputs, outputs) {
    return outputs[0].dtype;
  },

  createPipeline(device, _node, _inputs, outputs) {
    const dtype = outputs[0].dtype;
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: renderWgsl(source, dtype),
          label: `ConcatShader_${dtype}`,
        }),
        entryPoint: 'main',
      },
      label: `ConcatPipeline_${dtype}`,
    });
  },

  // Required by the interface but unused when `executeOverride` is set.
  buildBindGroupEntries() {
    throw new Error('Concat uses executeOverride; buildBindGroupEntries is not called');
  },
  getDispatch() {
    throw new Error('Concat uses executeOverride; getDispatch is not called');
  },

  executeOverride(device, encoder, node, inputs, outputs, pipeline) {
    const axis = node.attributes!.axis as number;
    const outMetaBuf = createMetaBuffer(device, packMeta(outputs[0]));
    const tempBuffers: GPUBuffer[] = [outMetaBuf];

    let axisStart = 0;
    for (const input of inputs) {
      const inMetaBuf = createMetaBuffer(device, packMeta(input));
      const concatData = new Uint32Array([axis, axisStart, 0, 0]);
      const concatBuf = createUniformBuffer(device, concatData);
      tempBuffers.push(inMetaBuf, concatBuf);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: input.storage.buffer as GPUBuffer } },
          { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
          { binding: 2, resource: { buffer: inMetaBuf } },
          { binding: 3, resource: { buffer: outMetaBuf } },
          { binding: 4, resource: { buffer: concatBuf } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(getShapeSize(input.shape) / 64), 1, 1);
      pass.end();

      axisStart += (input.shape as number[])[axis];
    }

    return { tempBuffers };
  },
};
