import source from './add.wgsl?raw';

export const getAddPipeline = (device: GPUDevice): GPUComputePipeline => {
  const shaderModule = device.createShaderModule({ code: source, label: 'AddShader' });
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
    label: 'AddPipeline'
  });
};
