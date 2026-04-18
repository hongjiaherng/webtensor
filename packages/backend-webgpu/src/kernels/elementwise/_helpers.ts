import { WebGPUKernel, packMeta, createMetaBuffer, getShapeSize, renderWgsl } from '../utils';

/**
 * Shared binary bind-group layout (a, b, out, meta_a, meta_b) and dispatch
 * (one thread per output element). The output dtype keys the pipeline cache
 * because `SCALAR` is substituted into the shader per dtype.
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

/**
 * Factory for element-wise broadcast comparisons. Same bindings as
 * `binaryKernel`; output is hardcoded `array<u32>` (bool). Inputs share one
 * arithmetic dtype (f32 / i32), so the pipeline cache keys on the input dtype.
 */
export function compareKernel(opLabel: string, source: string, pred: string): WebGPUKernel {
  return {
    pipelineKey(_node, inputs) {
      return inputs[0].dtype;
    },

    createPipeline(device, _node, inputs) {
      const inDType = inputs[0].dtype;
      const code = renderWgsl(source, inDType).replace(/\bPRED\b/g, pred);
      return device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: device.createShaderModule({
            code,
            label: `${opLabel}Shader_${inDType}`,
          }),
          entryPoint: 'main',
        },
        label: `${opLabel}Pipeline_${inDType}`,
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
