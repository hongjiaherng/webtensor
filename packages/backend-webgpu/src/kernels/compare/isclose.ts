import source from './isclose.wgsl';
import {
  WebGPUKernel,
  packMeta,
  createMetaBuffer,
  createUniformBuffer,
  getShapeSize,
  injectMeta,
} from '../utils';

/**
 * Float32-only tolerant equality. Extra uniform at binding 5 carries
 * `rtol` / `atol` / `equal_nan` (see compare/isclose.wgsl). Bindings mirror
 * the binary layout (a, b, out, meta_a, meta_b) plus one.
 */
export const iscloseKernel: WebGPUKernel = {
  pipelineKey() {
    return 'float32';
  },

  createPipeline(device) {
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: injectMeta(source),
          label: 'IsCloseShader',
        }),
        entryPoint: 'main',
      },
      label: 'IsClosePipeline',
    });
  },

  buildBindGroupEntries(device, node, inputs, outputs) {
    const outShape = outputs[0].shape as number[];
    const metaABuf = createMetaBuffer(device, packMeta(inputs[0], outShape));
    const metaBBuf = createMetaBuffer(device, packMeta(inputs[1], outShape));

    const rtol = node.attributes?.rtol as number;
    const atol = node.attributes?.atol as number;
    const equalNan = (node.attributes?.equalNan as number) ?? 0;

    const tolData = new Uint32Array(4);
    const f32View = new Float32Array(tolData.buffer);
    f32View[0] = rtol;
    f32View[1] = atol;
    tolData[2] = equalNan;
    tolData[3] = 0;
    const tolBuf = createUniformBuffer(device, tolData);

    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: inputs[1].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 3, resource: { buffer: metaABuf } },
        { binding: 4, resource: { buffer: metaBBuf } },
        { binding: 5, resource: { buffer: tolBuf } },
      ],
      tempBuffers: [metaABuf, metaBBuf, tolBuf],
    };
  },

  getDispatch(_node, _inputs, outputs) {
    return [Math.ceil(getShapeSize(outputs[0].shape) / 64), 1, 1];
  },
};
