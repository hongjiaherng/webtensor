import { Node } from '@minitensor/ir';
import { RuntimeTensor } from '@minitensor/runtime';
import { executeAdd } from './elementwise/add';
import { executeSub } from './elementwise/sub';
import { executeMul } from './elementwise/mul';
import { executeDiv } from './elementwise/div';
import { executeRelu, executeReluGrad } from './elementwise/relu';
import { executeMatMul } from './linear/matmul';
import { executeTranspose } from './shape/transpose';

export type CPUKernel = (node: Node, inputs: RuntimeTensor[], outputs: RuntimeTensor[]) => void;

const binaryElementwise = (
  fn: (a: Float32Array, b: Float32Array, out: Float32Array) => void
): CPUKernel => {
  return (_node, inputs, outputs) => {
    fn(
      inputs[0].buffer as Float32Array,
      inputs[1].buffer as Float32Array,
      outputs[0].buffer as Float32Array
    );
  };
};

export const cpuKernelRegistry = new Map<string, CPUKernel>([
  ['Add', binaryElementwise(executeAdd)],
  ['Sub', binaryElementwise(executeSub)],
  ['Mul', binaryElementwise(executeMul)],
  ['Div', binaryElementwise(executeDiv)],
  ['MatMul', (_node, inputs, outputs) => {
    const shapeA = inputs[0].shape as number[];
    const shapeB = inputs[1].shape as number[];
    const m = shapeA[shapeA.length - 2] || 1;
    const k = shapeA[shapeA.length - 1];
    const n = shapeB[shapeB.length - 1];

    executeMatMul(
      inputs[0].buffer as Float32Array,
      inputs[1].buffer as Float32Array,
      outputs[0].buffer as Float32Array,
      m,
      k,
      n
    );
  }],
  ['Relu', (_node, inputs, outputs) => {
    executeRelu(
      inputs[0].buffer as Float32Array,
      outputs[0].buffer as Float32Array
    );
  }],
  ['ReluGrad', (_node, inputs, outputs) => {
    executeReluGrad(
      inputs[0].buffer as Float32Array, // grad
      inputs[1].buffer as Float32Array, // original input a
      outputs[0].buffer as Float32Array
    );
  }],
  ['Transpose', (_node, inputs, outputs) => {
    const shape = inputs[0].shape as number[];
    const m = shape[shape.length - 2] || 1;
    const n = shape[shape.length - 1];

    executeTranspose(
      inputs[0].buffer as Float32Array,
      outputs[0].buffer as Float32Array,
      m,
      n
    );
  }],
]);
