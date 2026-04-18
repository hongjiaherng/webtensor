import { WebGPUKernel, packMeta, createMetaBuffer, getShapeSize, renderWgsl } from '../utils';

/**
 * All binary kernels share the same bind-group layout (a, b, out, meta_a, meta_b)
 * and the same dispatch (one thread per output element). They differ only by:
 *   - the WGSL source template
 *   - the label (debug name)
 *
 * Each kernel compiles a distinct pipeline per output dtype (SCALAR is
 * substituted in the shader) — `pipelineKey` keys the cache on dtype.
 */
export function binaryKernel(opLabel: string, source: string): WebGPUKernel {
  return {
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
            label: `${opLabel}Shader_${dtype}`,
          }),
          entryPoint: 'main',
        },
        label: `${opLabel}Pipeline_${dtype}`,
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
}
