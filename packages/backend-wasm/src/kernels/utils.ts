import { Node } from '@minitensor/ir';
import {
  RuntimeTensor,
  computeContiguousStrides,
  stridedIdx,
  isContiguous,
  broadcastStridesOf,
} from '@minitensor/runtime';
import { MinitensorWasmModule, WasmTensorHandle, isWasmTensorHandle } from '../module';

export { computeContiguousStrides, stridedIdx, isContiguous };

export function getShapeSize(shape: (number | null)[]): number {
  let size = 1;
  for (const dim of shape) {
    if (dim === null) throw new Error('Dynamic dimensions not yet supported in WASM backend');
    size *= dim;
  }
  return size;
}

export type WASMKernel = (
  module: MinitensorWasmModule,
  node: Node,
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[],
) => void;

export function handleOf(tensor: RuntimeTensor): WasmTensorHandle {
  if (!isWasmTensorHandle(tensor.storage.buffer)) {
    throw new Error('WASMBackend: expected a WASM tensor handle');
  }
  return tensor.storage.buffer;
}

// ---------------------------------------------------------------------------
// Meta buffer helpers

/**
 * Allocate a u32 meta block in WASM linear memory, fill it with `data`,
 * and return the pointer. The caller must call `module.free_u32(ptr, len)`.
 */
export function allocMeta(module: MinitensorWasmModule, data: Uint32Array): number {
  const ptr = module.alloc_u32(data.length);
  new Uint32Array(module.memory.buffer, ptr, data.length).set(data);
  return ptr;
}

/**
 * Build a 28-u32 meta block for binary elementwise ops.
 * Layout matches the Rust kernel / WebGPU design exactly:
 *   [0]      total
 *   [1]      rank
 *   [2..9]   out_shape[0..7]
 *   [10..17] a_broadcast_strides[0..7]
 *   [18]     a_offset
 *   [19..26] b_broadcast_strides[0..7]
 *   [27]     b_offset
 */
export function buildBinaryMetaData(
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[],
): Uint32Array {
  const outShape = outputs[0].shape as number[];
  const aShape = inputs[0].shape as number[];
  const bShape = inputs[1].shape as number[];
  const total = getShapeSize(outShape);
  const aBcast = broadcastStridesOf(outShape, aShape, inputs[0].strides);
  const bBcast = broadcastStridesOf(outShape, bShape, inputs[1].strides);

  const data = new Uint32Array(28);
  data[0] = total;
  data[1] = outShape.length;
  for (let i = 0; i < outShape.length; i++) {
    data[2 + i] = outShape[i];
    data[10 + i] = aBcast[i];
    data[19 + i] = bBcast[i];
  }
  data[18] = inputs[0].offset;
  data[27] = inputs[1].offset;
  return data;
}

/**
 * Build a 19-u32 meta block for unary elementwise ops.
 * Layout:
 *   [0]      total
 *   [1]      rank
 *   [2..9]   shape[0..7]
 *   [10..17] strides[0..7]
 *   [18]     offset
 */
export function buildUnaryMetaData(inputs: RuntimeTensor[]): Uint32Array {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  const total = getShapeSize(shape);

  const data = new Uint32Array(19);
  data[0] = total;
  data[1] = shape.length;
  for (let i = 0; i < shape.length; i++) {
    data[2 + i] = shape[i];
    data[10 + i] = strides[i];
  }
  data[18] = inputs[0].offset;
  return data;
}

/**
 * Build a 9-u32 meta block for matmul.
 * Layout:
 *   [0]  M   [1]  K   [2]  N
 *   [3]  a_row_stride   [4]  a_col_stride
 *   [5]  b_row_stride   [6]  b_col_stride
 *   [7]  a_offset       [8]  b_offset
 */
export function buildMatmulMetaData(inputs: RuntimeTensor[]): Uint32Array {
  const shapeA = inputs[0].shape as number[];
  const shapeB = inputs[1].shape as number[];
  const M = shapeA[shapeA.length - 2] ?? 1;
  const K = shapeA[shapeA.length - 1];
  const N = shapeB[shapeB.length - 1];
  const aStrides = inputs[0].strides;
  const bStrides = inputs[1].strides;

  const data = new Uint32Array(9);
  data[0] = M;
  data[1] = K;
  data[2] = N;
  data[3] = aStrides[aStrides.length - 2] ?? K;
  data[4] = aStrides[aStrides.length - 1];
  data[5] = bStrides[bStrides.length - 2] ?? N;
  data[6] = bStrides[bStrides.length - 1];
  data[7] = inputs[0].offset;
  data[8] = inputs[1].offset;
  return data;
}

/**
 * Build a 5-u32 meta block for transpose.
 * Layout:
 *   [0]  M   [1]  N   [2]  row_stride   [3]  col_stride   [4]  offset
 */
export function buildTransposeMetaData(inputs: RuntimeTensor[]): Uint32Array {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  const M = shape[shape.length - 2] ?? 1;
  const N = shape[shape.length - 1];

  const data = new Uint32Array(5);
  data[0] = M;
  data[1] = N;
  data[2] = strides[strides.length - 2] ?? N;
  data[3] = strides[strides.length - 1];
  data[4] = inputs[0].offset;
  return data;
}
