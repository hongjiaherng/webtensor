import source from './relu.wgsl?raw';

export const getReluPipeline = (device: GPUDevice): GPUComputePipeline => {
  const shaderModule = device.createShaderModule({ code: source, label: 'ReluShader' });
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
    label: 'ReluPipeline'
  });
};
