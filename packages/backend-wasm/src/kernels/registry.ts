import { Node } from '@minitensor/ir';
import { RuntimeTensor } from '@minitensor/runtime';
import { MinitensorWasmModule, WasmTensorHandle, isWasmTensorHandle } from '../module';

export type WASMKernel = (
  module: MinitensorWasmModule,
  node: Node,
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[]
) => void;

function handleOf(tensor: RuntimeTensor): WasmTensorHandle {
  if (!isWasmTensorHandle(tensor.buffer)) {
    throw new Error('WASMBackend: expected a WASM tensor handle');
  }
  return tensor.buffer;
}

const binaryElementwise = (
  fn: (module: MinitensorWasmModule, aPtr: number, bPtr: number, outPtr: number, lenA: number, lenB: number, lenOut: number) => void
): WASMKernel => {
  return (module, _node, inputs, outputs) => {
    const a = handleOf(inputs[0]);
    const b = handleOf(inputs[1]);
    const out = handleOf(outputs[0]);
    fn(module, a.ptr, b.ptr, out.ptr, a.elements, b.elements, out.elements);
  };
};

export const wasmKernelRegistry = new Map<string, WASMKernel>([
  ['Add', binaryElementwise((module, a, b, out, lenA, lenB, lenOut) => module.add_raw(a, b, out, lenA, lenB, lenOut))],
  ['Sub', binaryElementwise((module, a, b, out, lenA, lenB, lenOut) => module.sub_raw(a, b, out, lenA, lenB, lenOut))],
  ['Mul', binaryElementwise((module, a, b, out, lenA, lenB, lenOut) => module.mul_raw(a, b, out, lenA, lenB, lenOut))],
  ['Div', binaryElementwise((module, a, b, out, lenA, lenB, lenOut) => module.div_raw(a, b, out, lenA, lenB, lenOut))],
  ['MatMul', (module, _node, inputs, outputs) => {
    const shapeA = inputs[0].shape as number[];
    const shapeB = inputs[1].shape as number[];
    const m = shapeA[shapeA.length - 2] || 1;
    const k = shapeA[shapeA.length - 1];
    const n = shapeB[shapeB.length - 1];

    module.matmul_raw(
      handleOf(inputs[0]).ptr,
      handleOf(inputs[1]).ptr,
      handleOf(outputs[0]).ptr,
      m,
      k,
      n
    );
  }],
  ['Relu', (module, _node, inputs, outputs) => {
    const a = handleOf(inputs[0]);
    const out = handleOf(outputs[0]);
    module.relu_raw(a.ptr, out.ptr, out.elements);
  }],
  ['ReluGrad', (module, _node, inputs, outputs) => {
    const grad = handleOf(inputs[0]);
    const a = handleOf(inputs[1]);
    const out = handleOf(outputs[0]);
    module.relu_grad_raw(grad.ptr, a.ptr, out.ptr, out.elements);
  }],
  ['Transpose', (module, _node, inputs, outputs) => {
    const shape = inputs[0].shape as number[];
    const m = shape[shape.length - 2] || 1;
    const n = shape[shape.length - 1];

    module.transpose_raw(
      handleOf(inputs[0]).ptr,
      handleOf(outputs[0]).ptr,
      m,
      n
    );
  }],
]);
