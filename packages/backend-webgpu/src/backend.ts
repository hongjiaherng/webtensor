import { Node } from '@minitensor/ir';
import { Backend, RuntimeTensor } from '@minitensor/runtime';
import { getShapeSize, computeContiguousStrides } from './kernels/utils';
import { webgpuKernelRegistry, WebGPUKernel } from './kernels/registry';

export class WebGPUBackend implements Backend {
  private device: GPUDevice;
  private pipelineCache = new Map<string, GPUComputePipeline>();

  private constructor(device: GPUDevice) {
    this.device = device;
  }

  static async create(): Promise<WebGPUBackend> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error("WebGPU is not supported (navigator.gpu is missing).");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("No appropriate WebGPU adapter found");
    }
    const device = await adapter.requestDevice();
    return new WebGPUBackend(device);
  }

  allocate(shape: (number | null)[], dtype: 'float32' | 'int32' | 'bool'): RuntimeTensor {
    if (dtype !== 'float32') {
      throw new Error(`WebGPUBackend: unsupported dtype '${dtype}' — only float32 is currently implemented`);
    }
    const size = getShapeSize(shape);
    const byteSize = Math.ceil(size * 4 / 4) * 4;

    const gpuBuffer = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    return {
      storage: { buffer: gpuBuffer, byteLength: byteSize },
      shape,
      strides: computeContiguousStrides(shape as number[]),
      offset: 0,
      dtype,
    };
  }

  async read(tensor: RuntimeTensor): Promise<ArrayBufferView> {
    const srcBuffer = tensor.storage.buffer as GPUBuffer;

    const stagingBuffer = this.device.createBuffer({
      size: srcBuffer.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(srcBuffer, 0, stagingBuffer, 0, srcBuffer.size);
    this.device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange();

    const size = getShapeSize(tensor.shape);
    const view = new Float32Array(arrayBuffer.slice(0, size * 4));

    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return view;
  }

  write(tensor: RuntimeTensor, data: ArrayBufferView): void {
    const destBuffer = tensor.storage.buffer as GPUBuffer;
    this.device.queue.writeBuffer(destBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }

  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void {
    const kernel = this.getKernel(node.op);
    const pipeline = this.getPipeline(node.op, kernel);

    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();

    computePass.setPipeline(pipeline);

    const entries = kernel.buildBindGroupEntries(this.device, node, inputs, outputs);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    computePass.setBindGroup(0, bindGroup);

    const [x, y, z] = kernel.getDispatch(node, inputs, outputs);
    computePass.dispatchWorkgroups(x, y, z);
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  dispose(tensor: RuntimeTensor): void {
    if (tensor.storage.buffer) {
      const bufferToDestroy = tensor.storage.buffer as GPUBuffer;
      this.device.queue.onSubmittedWorkDone().then(() => {
        bufferToDestroy.destroy();
      });
      tensor.storage.buffer = null;
    }
  }

  private getKernel(op: string): WebGPUKernel {
    const kernel = webgpuKernelRegistry.get(op);
    if (!kernel) throw new Error(`WebGPUBackend: unsupported op '${op}'`);
    return kernel;
  }

  private getPipeline(op: string, kernel: WebGPUKernel): GPUComputePipeline {
    if (this.pipelineCache.has(op)) return this.pipelineCache.get(op)!;
    const pipeline = kernel.createPipeline(this.device);
    this.pipelineCache.set(op, pipeline);
    return pipeline;
  }
}
