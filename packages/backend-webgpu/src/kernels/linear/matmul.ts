import source from './matmul.wgsl?raw';

export const getMatMulPipeline = (device: GPUDevice): GPUComputePipeline => {
  const shaderModule = device.createShaderModule({ code: source, label: 'MatMulShader' });
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
    label: 'MatMulPipeline'
  });
};
