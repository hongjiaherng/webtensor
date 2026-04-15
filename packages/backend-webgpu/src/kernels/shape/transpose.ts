import transposeWGSL from './transpose.wgsl?raw';

export function getTransposePipeline(device: GPUDevice): GPUComputePipeline {
  const module = device.createShaderModule({
    label: 'Transpose Shader',
    code: transposeWGSL,
  });

  return device.createComputePipeline({
    label: 'Transpose Pipeline',
    layout: 'auto',
    compute: {
      module,
      entryPoint: 'main',
    },
  });
}
