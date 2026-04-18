import { Node, computeContiguousStrides } from '@webtensor/ir';
import {
  RuntimeTensor,
  getShapeSize,
  stridedIdx,
  isContiguous,
  broadcastStridesOf,
} from '@webtensor/runtime';
import { WebtensorWasmModule, WasmTensorHandle, isWasmTensorHandle } from '../module';

export { computeContiguousStrides, getShapeSize, stridedIdx, isContiguous };

export type WASMKernel = (
  module: WebtensorWasmModule,
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
export function allocMeta(module: WebtensorWasmModule, data: Uint32Array): number {
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
 * Build a 32-u32 meta block for batched matmul.
 * Layout:
 *   [0]       batch_rank
 *   [1]  M    [2]  K    [3]  N
 *   [4]  a_row_stride   [5]  a_col_stride
 *   [6]  b_row_stride   [7]  b_col_stride
 *   [8]  a_offset       [9]  b_offset
 *   [10..16]  batch_out_shape[0..6]
 *   [16..22]  a_bcast_batch_strides[0..6]
 *   [22..28]  b_bcast_batch_strides[0..6]
 *   [28..32]  padding
 * Max batch rank = 6 (total rank ≤ 8).
 */
export function buildMatmulMetaData(
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[],
): Uint32Array {
  const shapeA = inputs[0].shape as number[];
  const shapeB = inputs[1].shape as number[];
  const outShape = outputs[0].shape as number[];
  const rankA = shapeA.length;
  const rankB = shapeB.length;
  const outRank = outShape.length;
  const batchRank = outRank - 2;
  if (batchRank > 6) {
    throw new Error(`matmul: batch rank ${batchRank} exceeds WASM kernel cap of 6`);
  }

  const M = shapeA[rankA - 2];
  const K = shapeA[rankA - 1];
  const N = shapeB[rankB - 1];
  const aStrides = inputs[0].strides;
  const bStrides = inputs[1].strides;

  const batchOutShape = outShape.slice(0, batchRank);
  const aBatchShape = shapeA.slice(0, rankA - 2);
  const bBatchShape = shapeB.slice(0, rankB - 2);
  const aBatchStrides = aStrides.slice(0, rankA - 2);
  const bBatchStrides = bStrides.slice(0, rankB - 2);
  const aBcast =
    batchRank === 0 ? [] : broadcastStridesOf(batchOutShape, aBatchShape, aBatchStrides);
  const bBcast =
    batchRank === 0 ? [] : broadcastStridesOf(batchOutShape, bBatchShape, bBatchStrides);

  const data = new Uint32Array(32);
  data[0] = batchRank;
  data[1] = M;
  data[2] = K;
  data[3] = N;
  data[4] = aStrides[rankA - 2];
  data[5] = aStrides[rankA - 1];
  data[6] = bStrides[rankB - 2];
  data[7] = bStrides[rankB - 1];
  data[8] = inputs[0].offset;
  data[9] = inputs[1].offset;
  for (let i = 0; i < batchRank; i++) {
    data[10 + i] = batchOutShape[i];
    data[16 + i] = aBcast[i];
    data[22 + i] = bBcast[i];
  }
  return data;
}

/**
 * Build a 27-u32 meta block for reduction ops (ReduceSum, ReduceMean).
 * Layout:
 *   [0]       in_rank
 *   [1]       reduce_rank
 *   [2]       offset
 *   [3..11]   in_shape[0..8]
 *   [11..19]  in_strides[0..8]
 *   [19..27]  axes[0..8]  (only first reduce_rank valid)
 */
export function buildReduceMetaData(inputs: RuntimeTensor[], axes: number[]): Uint32Array {
  const inShape = inputs[0].shape as number[];
  const inStrides = inputs[0].strides;
  if (inShape.length > 8) {
    throw new Error(`reduce: rank ${inShape.length} exceeds WASM kernel cap of 8`);
  }

  const data = new Uint32Array(27);
  data[0] = inShape.length;
  data[1] = axes.length;
  data[2] = inputs[0].offset;
  for (let i = 0; i < inShape.length; i++) {
    data[3 + i] = inShape[i];
    data[11 + i] = inStrides[i];
  }
  for (let i = 0; i < axes.length; i++) {
    data[19 + i] = axes[i];
  }
  return data;
}

/**
 * Build a 19-u32 meta block for softmax.
 * Layout:
 *   [0]       rank
 *   [1]       axis
 *   [2]       offset
 *   [3..11]   shape[0..8]
 *   [11..19]  strides[0..8]
 */
export function buildSoftmaxMetaData(inputs: RuntimeTensor[], axis: number): Uint32Array {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  if (shape.length > 8) {
    throw new Error(`softmax: rank ${shape.length} exceeds WASM kernel cap of 8`);
  }

  const data = new Uint32Array(19);
  data[0] = shape.length;
  data[1] = axis;
  data[2] = inputs[0].offset;
  for (let i = 0; i < shape.length; i++) {
    data[3 + i] = shape[i];
    data[11 + i] = strides[i];
  }
  return data;
}

/** Dtype suffix for function name lookups (`_f32_`, `_i32_`, `_u8_`). Dtype-ready scaffolding. */
export function dtypeSuffix(dtype: import('@webtensor/ir').DType): 'f32' | 'i32' | 'u8' {
  switch (dtype) {
    case 'float32':
      return 'f32';
    case 'int32':
      return 'i32';
    case 'bool':
      return 'u8';
  }
}
