import { Node } from '@minitensor/ir';
import { Backend, RuntimeTensor } from '@minitensor/runtime';
import { getShapeSize } from './kernels/utils';
import { webgpuKernelRegistry } from './kernels/registry';

export class WebGPUBackend implements Backend {
  private device: GPUDevice;
  private pipelineCache = new Map<string, GPUComputePipeline>();
  
  private constructor(device: GPUDevice) {
    this.device = device;
  }

  static async create(): Promise<WebGPUBackend> {
    // navigator.gpu is attached to the global scope in browser or explicitly bound
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
    const size = getShapeSize(shape);
    let byteSize = size * 4; // Assuming f32/i32 for MVP
    byteSize = Math.ceil(byteSize / 4) * 4;
    
    const buffer = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    return { shape, dtype, buffer };
  }

  async read(tensor: RuntimeTensor): Promise<ArrayBufferView> {
    const srcBuffer = tensor.buffer as GPUBuffer;
    
    // Create staging buffer to be mapped safely locally
    const stagingBuffer = this.device.createBuffer({
      size: srcBuffer.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Schedule the command to duplicate memory from Compute -> CPU Staging
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(srcBuffer, 0, stagingBuffer, 0, srcBuffer.size);
    this.device.queue.submit([commandEncoder.finish()]);

    // Await actual hardware synchronization block natively
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = stagingBuffer.getMappedRange();
    
    const size = getShapeSize(tensor.shape);
    let view: ArrayBufferView;
    if (tensor.dtype === 'float32') {
      view = new Float32Array(arrayBuffer.slice(0, size * 4));
    } else {
      view = new Int32Array(arrayBuffer.slice(0, size * 4));
    }
    
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    
    return view;
  }

  write(tensor: RuntimeTensor, data: ArrayBufferView): void {
    const destBuffer = tensor.buffer as GPUBuffer;
    this.device.queue.writeBuffer(destBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }

  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void {
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();

    const pipeline = this.getPipeline(node.op);
    computePass.setPipeline(pipeline);

    const bindGroupEntries: GPUBindGroupEntry[] = [];
    for (let i = 0; i < inputs.length; i++) {
      bindGroupEntries.push({
        binding: i,
        resource: { buffer: inputs[i].buffer as GPUBuffer }
      });
    }

    // Bind output
    bindGroupEntries.push({
      binding: inputs.length,
      resource: { buffer: outputs[0].buffer as GPUBuffer }
    });

    // Handle MatMul explicit 2D dimension mapping configuration via UBO injection
    if (node.op === 'MatMul') {
      const shapeA = inputs[0].shape as number[];
      const shapeB = inputs[1].shape as number[];
      const M = shapeA[shapeA.length - 2] || 1;
      const K = shapeA[shapeA.length - 1];
      const N = shapeB[shapeB.length - 1];
      
      const localMeta = this.device.createBuffer({
        size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true
      });
      new Uint32Array(localMeta.getMappedRange()).set([M, K, N, 0]);
      localMeta.unmap();
      
      bindGroupEntries.push({
        binding: inputs.length + 1, // index 3
        resource: { buffer: localMeta }
      });
    } else if (node.op === 'Transpose') {
      const shape = inputs[0].shape as number[];
      const M = shape[shape.length - 2] || 1;
      const N = shape[shape.length - 1];
      
      const localMeta = this.device.createBuffer({
        size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true
      });
      new Uint32Array(localMeta.getMappedRange()).set([M, N, 0, 0]);
      localMeta.unmap();
      
      bindGroupEntries.push({
        binding: inputs.length + 1, // index 2
        resource: { buffer: localMeta }
      });
    }

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries
    });

    computePass.setBindGroup(0, bindGroup);

    // Compute invocation topology specifically mapped for the algorithm geometry structurally
    if (node.op === 'MatMul') {
      const shapeA = inputs[0].shape as number[];
      const shapeB = inputs[1].shape as number[];
      const M = shapeA[shapeA.length - 2] || 1;
      const N = shapeB[shapeB.length - 1];
      // We map directly onto the X and Y axes using our 8x8 thread workgroup grid layout!
      computePass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(N / 8));
    } else if (node.op === 'Transpose') {
      const outElements = getShapeSize(outputs[0].shape);
      const workgroups = Math.ceil(outElements / 64);
      computePass.dispatchWorkgroups(workgroups);
    } else {
      // Element-wise inherently spans a pure 1D X axis logically scaling out flat elements directly
      const outElements = getShapeSize(outputs[0].shape);
      const workgroups = Math.ceil(outElements / 64);
      computePass.dispatchWorkgroups(workgroups);
    }
    
    computePass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  dispose(tensor: RuntimeTensor): void {
    if (tensor.buffer) {
       const bufferToDestroy = tensor.buffer as GPUBuffer;
       this.device.queue.onSubmittedWorkDone().then(() => {
         bufferToDestroy.destroy();
       });
       tensor.buffer = null;
    }
  }

  private getPipeline(op: string): GPUComputePipeline {
    if (this.pipelineCache.has(op)) {
      return this.pipelineCache.get(op)!;
    }

    const factory = webgpuKernelRegistry.get(op);
    if (!factory) throw new Error(`Unsupported WebGPU op: ${op}`);
    const pipeline = factory(this.device);

    this.pipelineCache.set(op, pipeline);
    return pipeline;
  }
}
