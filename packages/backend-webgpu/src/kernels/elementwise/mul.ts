import source from './mul.wgsl?raw';

export const getMulPipeline = (device: GPUDevice): GPUComputePipeline => {
  const shaderModule = device.createShaderModule({ code: source, label: 'MulShader' });
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
    label: 'MulPipeline'
  });
};
