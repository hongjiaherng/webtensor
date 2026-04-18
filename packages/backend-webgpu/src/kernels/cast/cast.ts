import { DType } from '@webtensor/ir';
import source from './cast.wgsl';
import { WebGPUKernel, packMeta, createMetaBuffer, getShapeSize, injectMeta } from '../utils';

const WGSL_SCALAR: Record<DType, string> = {
  float32: 'f32',
  int32: 'i32',
  bool: 'u32',
};

function assertSupported(dtype: DType, side: 'in' | 'out'): void {
  if (dtype === 'bool') {
    throw new Error(
      `WebGPU cast: bool is not supported as the ${side === 'in' ? 'input' : 'output'} dtype. ` +
        'Cast bool tensors on CPU or WASM first.',
    );
  }
}

/**
 * Cast kernel. Compiles one pipeline per (in_dtype, out_dtype) pair by doing
 * two substitutions (IN_SCALAR, OUT_SCALAR) in the WGSL template. The engine's
 * pipeline cache keys on `Cast:{key}` so each pair is compiled once per backend.
 */
export const castKernel: WebGPUKernel = {
  pipelineKey(_node, inputs, outputs) {
    return `${inputs[0].dtype}-${outputs[0].dtype}`;
  },

  createPipeline(device, _node, inputs, outputs) {
    const inDType = inputs[0].dtype;
    const outDType = outputs[0].dtype;
    assertSupported(inDType, 'in');
    assertSupported(outDType, 'out');
    const code = injectMeta(source)
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
