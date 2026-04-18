import source from './cast.wgsl';
import {
  WebGPUKernel,
  packMeta,
  createMetaBuffer,
  getShapeSize,
  injectMeta,
  WGSL_SCALAR,
} from '../utils';

/**
 * Cast kernel. Compiles one pipeline per (in_dtype, out_dtype) pair by
 * substituting IN_SCALAR / OUT_SCALAR / CAST_EXPR in the WGSL template. The
 * engine's pipeline cache keys on `Cast:{key}` so each pair is compiled once
 * per backend.
 */
export const castKernel: WebGPUKernel = {
  pipelineKey(_node, inputs, outputs) {
    return `${inputs[0].dtype}-${outputs[0].dtype}`;
  },

  createPipeline(device, _node, inputs, outputs) {
    const inDType = inputs[0].dtype;
    const outDType = outputs[0].dtype;
    const castExpr =
      outDType === 'bool' ? `select(0u, 1u, v != ${WGSL_SCALAR[inDType]}(0))` : `OUT_SCALAR(v)`;
    const code = injectMeta(source)
      .replace(/\bCAST_EXPR\b/g, castExpr)
      .replace(/\bIN_SCALAR\b/g, WGSL_SCALAR[inDType])
      .replace(/\bOUT_SCALAR\b/g, WGSL_SCALAR[outDType]);
    return device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code,
          label: `CastShader_${inDType}_${outDType}`,
        }),
        entryPoint: 'main',
      },
      label: `CastPipeline_${inDType}_${outDType}`,
    });
  },

  buildBindGroupEntries(device, _node, inputs, outputs) {
    const metaBuf = createMetaBuffer(device, packMeta(inputs[0]));
    return {
      entries: [
        { binding: 0, resource: { buffer: inputs[0].storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: outputs[0].storage.buffer as GPUBuffer } },
        { binding: 2, resource: { buffer: metaBuf } },
      ],
      tempBuffers: [metaBuf],
    };
  },

  getDispatch(_node, _inputs, outputs) {
    return [Math.ceil(getShapeSize(outputs[0].shape) / 64), 1, 1];
  },
};
