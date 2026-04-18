import { Node, computeContiguousStrides, MAX_RANK } from '@webtensor/ir';
import {
  RuntimeTensor,
  getShapeSize,
  stridedIdx,
  isContiguous,
  broadcastStridesOf,
} from '@webtensor/runtime';
import { WebtensorWasmModule, WasmTensorHandle, isWasmTensorHandle } from '../module';

export { computeContiguousStrides, getShapeSize, stridedIdx, isContiguous, MAX_RANK };

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
// Meta buffer layouts. Offsets and widths must stay in lockstep with the Rust
// constants in `rust/src/ops/mod.rs` — a single `MAX_RANK` drives both sides.

// Unary: [total, rank, shape[MAX_RANK], strides[MAX_RANK], offset]
const UNARY_META_WORDS = 3 + 2 * MAX_RANK;
const UNARY_SHAPE_OFF = 2;
const UNARY_STRIDES_OFF = 2 + MAX_RANK;
const UNARY_OFFSET_OFF = 2 + 2 * MAX_RANK;

// Binary: [total, rank, out_shape[MAX_RANK], a_strides[MAX_RANK], a_off,
//          b_strides[MAX_RANK], b_off]
const BINARY_META_WORDS = 4 + 3 * MAX_RANK;
const BINARY_SHAPE_OFF = 2;
const BINARY_A_STRIDES_OFF = 2 + MAX_RANK;
const BINARY_A_OFFSET_OFF = 2 + 2 * MAX_RANK;
const BINARY_B_STRIDES_OFF = 3 + 2 * MAX_RANK;
const BINARY_B_OFFSET_OFF = 3 + 3 * MAX_RANK;

// Reduce: [in_rank, reduce_rank, offset, in_shape[MAX_RANK],
//          in_strides[MAX_RANK], axes[MAX_RANK]]
const REDUCE_META_WORDS = 3 + 3 * MAX_RANK;
const REDUCE_SHAPE_OFF = 3;
const REDUCE_STRIDES_OFF = 3 + MAX_RANK;
const REDUCE_AXES_OFF = 3 + 2 * MAX_RANK;

// Softmax: [rank, axis, offset, shape[MAX_RANK], strides[MAX_RANK]]
const SOFTMAX_META_WORDS = 3 + 2 * MAX_RANK;
const SOFTMAX_SHAPE_OFF = 3;
const SOFTMAX_STRIDES_OFF = 3 + MAX_RANK;

// Matmul: [batch_rank, M, K, N, a_row_s, a_col_s, b_row_s, b_col_s, a_off,
//          b_off, batch_out_shape[MAX_RANK-2], a_bcast[MAX_RANK-2],
//          b_bcast[MAX_RANK-2]]
const MATMUL_BATCH_CAP = MAX_RANK - 2;
const MATMUL_META_WORDS = 10 + 3 * MATMUL_BATCH_CAP;
const MATMUL_BATCH_SHAPE_OFF = 10;
const MATMUL_A_BCAST_OFF = 10 + MATMUL_BATCH_CAP;
const MATMUL_B_BCAST_OFF = 10 + 2 * MATMUL_BATCH_CAP;

// Concat: [total, rank, in_shape[MAX_RANK], in_strides[MAX_RANK], in_offset,
//          out_shape[MAX_RANK], axis, axis_start]
const CONCAT_META_WORDS = 5 + 3 * MAX_RANK;
const CONCAT_IN_SHAPE_OFF = 2;
const CONCAT_IN_STRIDES_OFF = 2 + MAX_RANK;
const CONCAT_IN_OFFSET_OFF = 2 + 2 * MAX_RANK;
const CONCAT_OUT_SHAPE_OFF = 3 + 2 * MAX_RANK;
const CONCAT_AXIS_OFF = 3 + 3 * MAX_RANK;
const CONCAT_AXIS_START_OFF = 4 + 3 * MAX_RANK;

// Pad: [src_total, rank, src_shape[MAX_RANK], src_strides[MAX_RANK],
//       src_offset, out_shape[MAX_RANK], pads_before[MAX_RANK]]
const PAD_META_WORDS = 3 + 4 * MAX_RANK;
const PAD_SRC_SHAPE_OFF = 2;
const PAD_SRC_STRIDES_OFF = 2 + MAX_RANK;
const PAD_SRC_OFFSET_OFF = 2 + 2 * MAX_RANK;
const PAD_OUT_SHAPE_OFF = 3 + 2 * MAX_RANK;
const PAD_PADS_BEFORE_OFF = 3 + 3 * MAX_RANK;

/**
 * Allocate a u32 meta block in WASM linear memory, fill it with `data`,
 * and return the pointer. The caller must call `module.free_u32(ptr, len)`.
 */
export function allocMeta(module: WebtensorWasmModule, data: Uint32Array): number {
  const ptr = module.alloc_u32(data.length);
  new Uint32Array(module.memory.buffer, ptr, data.length).set(data);
  return ptr;
}

function assertRank(rank: number, label: string): void {
  if (rank > MAX_RANK) {
    throw new Error(`${label}: rank ${rank} exceeds MAX_RANK ${MAX_RANK}`);
  }
}

/**
 * Build a meta block for binary elementwise ops. Layout matches
 * `rust/src/ops/mod.rs::BINARY_META_WORDS`.
 */
export function buildBinaryMetaData(
  inputs: RuntimeTensor[],
  outputs: RuntimeTensor[],
): Uint32Array {
  const outShape = outputs[0].shape as number[];
  const aShape = inputs[0].shape as number[];
  const bShape = inputs[1].shape as number[];
  assertRank(outShape.length, 'binary');
  const total = getShapeSize(outShape);
  const aBcast = broadcastStridesOf(outShape, aShape, inputs[0].strides);
  const bBcast = broadcastStridesOf(outShape, bShape, inputs[1].strides);

  const data = new Uint32Array(BINARY_META_WORDS);
  data[0] = total;
  data[1] = outShape.length;
  for (let i = 0; i < outShape.length; i++) {
    data[BINARY_SHAPE_OFF + i] = outShape[i];
    data[BINARY_A_STRIDES_OFF + i] = aBcast[i];
    data[BINARY_B_STRIDES_OFF + i] = bBcast[i];
  }
  data[BINARY_A_OFFSET_OFF] = inputs[0].offset;
  data[BINARY_B_OFFSET_OFF] = inputs[1].offset;
  return data;
}

/**
 * Build a meta block for unary elementwise ops. Layout matches
 * `rust/src/ops/mod.rs::UNARY_META_WORDS`.
 */
export function buildUnaryMetaData(inputs: RuntimeTensor[]): Uint32Array {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  assertRank(shape.length, 'unary');
  const total = getShapeSize(shape);

  const data = new Uint32Array(UNARY_META_WORDS);
  data[0] = total;
  data[1] = shape.length;
  for (let i = 0; i < shape.length; i++) {
    data[UNARY_SHAPE_OFF + i] = shape[i];
    data[UNARY_STRIDES_OFF + i] = strides[i];
  }
  data[UNARY_OFFSET_OFF] = inputs[0].offset;
  return data;
}

/**
 * Build a meta block for batched matmul. Batch rank is capped at
 * `MAX_RANK - 2` so total tensor rank fits the shared cap. Layout matches
 * `rust/src/ops/mod.rs::MATMUL_META_WORDS`.
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
  if (batchRank > MATMUL_BATCH_CAP) {
    throw new Error(
      `matmul: batch rank ${batchRank} exceeds cap of ${MATMUL_BATCH_CAP} (MAX_RANK - 2)`,
    );
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

  const data = new Uint32Array(MATMUL_META_WORDS);
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
    data[MATMUL_BATCH_SHAPE_OFF + i] = batchOutShape[i];
    data[MATMUL_A_BCAST_OFF + i] = aBcast[i];
    data[MATMUL_B_BCAST_OFF + i] = bBcast[i];
  }
  return data;
}

/**
 * Build a meta block for reduction ops (ReduceSum, ReduceMean). Layout
 * matches `rust/src/ops/mod.rs::REDUCE_META_WORDS`.
 */
export function buildReduceMetaData(inputs: RuntimeTensor[], axes: number[]): Uint32Array {
  const inShape = inputs[0].shape as number[];
  const inStrides = inputs[0].strides;
  assertRank(inShape.length, 'reduce');

  const data = new Uint32Array(REDUCE_META_WORDS);
  data[0] = inShape.length;
  data[1] = axes.length;
  data[2] = inputs[0].offset;
  for (let i = 0; i < inShape.length; i++) {
    data[REDUCE_SHAPE_OFF + i] = inShape[i];
    data[REDUCE_STRIDES_OFF + i] = inStrides[i];
  }
  for (let i = 0; i < axes.length; i++) {
    data[REDUCE_AXES_OFF + i] = axes[i];
  }
  return data;
}

/**
 * Build a meta block for softmax. Layout matches
 * `rust/src/ops/mod.rs::SOFTMAX_META_WORDS`.
 */
export function buildSoftmaxMetaData(inputs: RuntimeTensor[], axis: number): Uint32Array {
  const shape = inputs[0].shape as number[];
  const strides = inputs[0].strides;
  assertRank(shape.length, 'softmax');

  const data = new Uint32Array(SOFTMAX_META_WORDS);
  data[0] = shape.length;
  data[1] = axis;
  data[2] = inputs[0].offset;
  for (let i = 0; i < shape.length; i++) {
    data[SOFTMAX_SHAPE_OFF + i] = shape[i];
    data[SOFTMAX_STRIDES_OFF + i] = strides[i];
  }
  return data;
}

/**
 * Build a meta block for one concat input. Layout matches
 * `rust/src/ops/mod.rs::CONCAT_META_WORDS`.
 */
export function buildConcatMetaData(
  input: RuntimeTensor,
  outShape: number[],
  axis: number,
  axisStart: number,
): Uint32Array {
  const inShape = input.shape as number[];
  const rank = inShape.length;
  assertRank(rank, 'concat');
  const total = getShapeSize(inShape);
  const data = new Uint32Array(CONCAT_META_WORDS);
  data[0] = total;
  data[1] = rank;
  for (let i = 0; i < rank; i++) {
    data[CONCAT_IN_SHAPE_OFF + i] = inShape[i];
    data[CONCAT_IN_STRIDES_OFF + i] = input.strides[i];
    data[CONCAT_OUT_SHAPE_OFF + i] = outShape[i];
  }
  data[CONCAT_IN_OFFSET_OFF] = input.offset;
  data[CONCAT_AXIS_OFF] = axis;
  data[CONCAT_AXIS_START_OFF] = axisStart;
  return data;
}

/**
 * Build a meta block for pad. Layout matches
 * `rust/src/ops/mod.rs::PAD_META_WORDS`.
 */
export function buildPadMetaData(
  src: RuntimeTensor,
  outShape: number[],
  padsBefore: number[],
): Uint32Array {
  const srcShape = src.shape as number[];
  const rank = srcShape.length;
  assertRank(rank, 'pad');
  const srcTotal = getShapeSize(srcShape);
  const data = new Uint32Array(PAD_META_WORDS);
  data[0] = srcTotal;
  data[1] = rank;
  for (let i = 0; i < rank; i++) {
    data[PAD_SRC_SHAPE_OFF + i] = srcShape[i];
    data[PAD_SRC_STRIDES_OFF + i] = src.strides[i];
    data[PAD_OUT_SHAPE_OFF + i] = outShape[i];
    data[PAD_PADS_BEFORE_OFF + i] = padsBefore[i];
  }
  data[PAD_SRC_OFFSET_OFF] = src.offset;
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
