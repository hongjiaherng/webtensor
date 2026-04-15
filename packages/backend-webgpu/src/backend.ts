import { Node } from '@minitensor/ir';
import { Backend, RuntimeTensor } from '@minitensor/runtime';
import { getShapeSize, computeContiguousStrides, isContiguous, alignTo } from './kernels/utils';
import { webgpuKernelRegistry, WebGPUKernel } from './kernels/registry';
import gatherSource from './kernels/internal/gather.wgsl?raw';

// Gather meta buffer layout (19 × u32 = 76 bytes, padded to 80):
//   [0]      total elements
//   [1]      rank
//   [2..9]   shape[0..7]
//   [10..17] strides[0..7]
//   [18]     src_offset
const GATHER_META_WORDS = 19;
const GATHER_META_BYTES = alignTo(GATHER_META_WORDS * 4, 16); // = 80

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
    const commandEncoder = this.device.createCommandEncoder();

    // Gather pre-pass: make any non-contiguous inputs contiguous.
    // Each gather runs in its own compute pass inside this encoder so that the
    // main kernel sees the packed data in the same queue submission.
    const contiguousInputs: RuntimeTensor[] = inputs.map(t => {
      const shape = t.shape as number[];
      if (isContiguous(shape, t.strides, t.offset)) return t;
      return this.gatherPass(commandEncoder, t);
    });

    // Main kernel compute pass
    const kernel = this.getKernel(node.op);
    const pipeline = this.getPipeline(node.op, kernel);
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    const entries = kernel.buildBindGroupEntries(this.device, node, contiguousInputs, outputs);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    computePass.setBindGroup(0, bindGroup);
    const [x, y, z] = kernel.getDispatch(node, contiguousInputs, outputs);
    computePass.dispatchWorkgroups(x, y, z);
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Destroy temporary contiguous buffers created by gatherPass.
    // WebGPU keeps internal references alive until the submitted work completes.
    this.device.queue.onSubmittedWorkDone().then(() => {
      contiguousInputs.forEach((t, i) => {
        if (t !== inputs[i]) {
          (t.storage.buffer as GPUBuffer).destroy();
        }
      });
    });
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

  // ---------------------------------------------------------------------------
  // Private helpers

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

  /**
   * Appends a gather compute pass to `enc` that reads the non-contiguous `src`
   * tensor with strided indexing and writes into a fresh contiguous GPUBuffer.
   * Returns a new RuntimeTensor wrapping the packed destination buffer.
   * The caller is responsible for destroying the returned tensor's buffer.
   */
  private gatherPass(enc: GPUCommandEncoder, src: RuntimeTensor): RuntimeTensor {
    const shape = src.shape as number[];
    const total = getShapeSize(shape);

    const dstBuffer = this.device.createBuffer({
      size: total * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Build meta storage buffer (not uniform — avoids 16-byte element padding)
    const metaData = new Uint32Array(GATHER_META_WORDS);
    metaData[0] = total;
    metaData[1] = shape.length;
    for (let i = 0; i < shape.length; i++) {
      metaData[2 + i]  = shape[i];
      metaData[10 + i] = src.strides[i];
    }
    metaData[18] = src.offset;

    const metaBuffer = this.device.createBuffer({
      size: GATHER_META_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(metaBuffer.getMappedRange()).set(metaData);
    metaBuffer.unmap();

    const gatherPipeline = this.getGatherPipeline();
    const bindGroup = this.device.createBindGroup({
      layout: gatherPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src.storage.buffer as GPUBuffer } },
        { binding: 1, resource: { buffer: dstBuffer } },
        { binding: 2, resource: { buffer: metaBuffer } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(gatherPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 64));
    pass.end();

    // Schedule meta buffer cleanup; GPU holds internal ref until work completes.
    this.device.queue.onSubmittedWorkDone().then(() => metaBuffer.destroy());

    return {
      storage: { buffer: dstBuffer, byteLength: total * 4 },
      shape,
      strides: computeContiguousStrides(shape),
      offset: 0,
      dtype: src.dtype,
    };
  }

  private getGatherPipeline(): GPUComputePipeline {
    const key = '_Gather';
    if (this.pipelineCache.has(key)) return this.pipelineCache.get(key)!;
    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({ code: gatherSource, label: 'GatherShader' }),
        entryPoint: 'main',
      },
      label: 'GatherPipeline',
    });
    this.pipelineCache.set(key, pipeline);
    return pipeline;
  }
}
