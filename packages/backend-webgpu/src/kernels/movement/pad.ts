import { MAX_RANK } from '@webtensor/ir';
import source from './pad.wgsl';
import {
  WebGPUKernel,
  packMeta,
  createMetaBuffer,
  createUniformBuffer,
  getShapeSize,
  renderWgsl,
} from '../utils';

/**
 * Constant-value Pad. Single dispatch, one thread per output element (see
 * pad.wgsl for the gather-style index math). Fill value is carried as raw
 * bits so the same shader works for f32 / i32 / u32 (bool).
 */
export const padKernel: WebGPUKernel = {
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
          label: `PadShader_${dtype}`,
        }),
        entryPoint: 'main',
      },
      label: `PadPipeline_${dtype}`,
    });
  },

  buildBindGroupEntries(device, node, inputs, outputs) {
    const pads = node.attributes!.pads as number[];
    const value = (node.attributes!.value as number) ?? 0;
    const dtype = outputs[0].dtype;

    const inMetaBuf = createMetaBuffer(device, packMeta(inputs[0]));
    const outMetaBuf = createMetaBuffer(device, packMeta(outputs[0]));

    // PadMeta: pads_before[MAX_RANK] u32 + value_bits u32 + 3 pad u32.
    const padData = new Uint32Array(MAX_RANK + 4);
    for (let i = 0; i < MAX_RANK; i++) padData[i] = i < pads.length ? pads[i] : 0;
    if (dtype === 'float32') {
      new Float32Array(padData.buffer, MAX_RANK * 4, 1)[0] = value;
    } else {
      padData[MAX_RANK] = value >>> 0;
    }
    const padBuf = createUniformBuffer(device, padData);

    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: inMetaBuf } },
        { binding: 3, resource: { buffer: outMetaBuf } },
        { binding: 4, resource: { buffer: padBuf } },
      ],
      tempBuffers: [inMetaBuf, outMetaBuf, padBuf],
    };
  },

  getDispatch(_node, _inputs, outputs) {
    return [Math.ceil(getShapeSize(outputs[0].shape) / 64), 1, 1];
  },
};
