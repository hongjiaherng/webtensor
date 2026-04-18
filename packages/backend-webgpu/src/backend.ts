import { Node, DType } from '@webtensor/ir';
import { Backend, RuntimeTensor, bytesPerElement, typedArrayCtor } from '@webtensor/runtime';
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
      throw new Error('WebGPU is not supported (navigator.gpu is missing).');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No appropriate WebGPU adapter found');
    }
    const device = await adapter.requestDevice();
    return new WebGPUBackend(device);
  }

  allocate(shape: (number | null)[], dtype: DType): RuntimeTensor {
    const size = getShapeSize(shape);
    // WebGPU storage buffers must be a multiple of 4 bytes — round up for bool (1 B/elem).
    const rawByteLength = size * bytesPerElement(dtype);
    const byteSize = Math.max(4, Math.ceil(rawByteLength / 4) * 4);

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
    const bpe = bytesPerElement(tensor.dtype);
    const Ctor = typedArrayCtor(tensor.dtype);
    const view = new Ctor(arrayBuffer.slice(0, size * bpe));

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
    const pipeline = this.getPipeline(node.op, kernel, node, inputs, outputs);

    const { entries, tempBuffers } = kernel.buildBindGroupEntries(
      this.device,
      node,
      inputs,
      outputs,
    );
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });

    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    const [x, y, z] = kernel.getDispatch(node, inputs, outputs);
    computePass.dispatchWorkgroups(x, y, z);
    computePass.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // Destroy ephemeral meta buffers once the GPU is done.
    // WebGPU keeps the underlying resources alive until all submitted work completes.
    if (tempBuffers.length > 0) {
      this.device.queue.onSubmittedWorkDone().then(() => {
        for (const buf of tempBuffers) buf.destroy();
      });
    }
  }

  dispose(tensor: RuntimeTensor): void {
    if (tensor.isView) return;
    if (tensor.storage.buffer) {
      const bufferToDestroy = tensor.storage.buffer as GPUBuffer;
      this.device.queue.onSubmittedWorkDone().then(() => {
        bufferToDestroy.destroy();
      });
      tensor.storage.buffer = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Private helpers

  private getKernel(op: string): WebGPUKernel {
    const kernel = webgpuKernelRegistry.get(op);
    if (!kernel) throw new Error(`WebGPUBackend: unsupported op '${op}'`);
    return kernel;
  }

  private getPipeline(
    op: string,
    kernel: WebGPUKernel,
    node: Node,
    inputs: RuntimeTensor[],
    outputs: RuntimeTensor[],
  ): GPUComputePipeline {
    // Kernels that vary by dtype (or any other build-time attribute) expose
    // a `pipelineKey`. The cache keys on op + key so multiple dtype variants
    // coexist per backend instance.
    const key = kernel.pipelineKey
      ? `${op}:${kernel.pipelineKey(node, inputs, outputs)}`
      : op;
    const cached = this.pipelineCache.get(key);
    if (cached) return cached;
    const pipeline = kernel.createPipeline(this.device, node, inputs, outputs);
    this.pipelineCache.set(key, pipeline);
    return pipeline;
  }
}
