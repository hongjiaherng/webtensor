import { Node } from '@minitensor/ir';

export interface RuntimeTensor {
  shape: (number | null)[];
  dtype: 'float32' | 'int32' | 'bool';
  // Abstract reference, could be `GPUBuffer` or `Float32Array`
  buffer: any;
}

export interface Backend {
  allocate(shape: (number | null)[], dtype: 'float32' | 'int32' | 'bool'): RuntimeTensor;
  read(tensor: RuntimeTensor): Promise<ArrayBufferView>;
  write(tensor: RuntimeTensor, data: ArrayBufferView): void;
  // Execution fills the pre-allocated output buffers
  execute(node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]): void;
  dispose(tensor: RuntimeTensor): void;
}
