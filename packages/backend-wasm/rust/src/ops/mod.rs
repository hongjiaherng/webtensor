pub mod activation;
pub mod binary;
pub mod cast;
pub mod compare;
pub mod join;
pub mod linalg;
pub mod padding;
pub mod reduce;
pub mod unary;

/// Maximum tensor rank supported by every WASM kernel. PyTorch-aligned.
/// All meta layouts size their fixed shape/strides/axis arrays to this bound;
/// bumping it widens every meta buffer and requires a matching TS side update
/// in `@webtensor/ir` and `packages/backend-wasm/src/kernels/utils.ts`.
pub const MAX_RANK: usize = 64;

/// Unary meta: `[total, rank, shape[MAX_RANK], strides[MAX_RANK], offset]`.
pub const UNARY_META_WORDS: usize = 3 + 2 * MAX_RANK;
pub const UNARY_SHAPE_OFF: usize = 2;
pub const UNARY_STRIDES_OFF: usize = 2 + MAX_RANK;
pub const UNARY_OFFSET_OFF: usize = 2 + 2 * MAX_RANK;

/// Binary meta:
/// `[total, rank, out_shape[MAX_RANK], a_strides[MAX_RANK], a_off,
///   b_strides[MAX_RANK], b_off]`.
pub const BINARY_META_WORDS: usize = 4 + 3 * MAX_RANK;
pub const BINARY_SHAPE_OFF: usize = 2;
pub const BINARY_A_STRIDES_OFF: usize = 2 + MAX_RANK;
pub const BINARY_A_OFFSET_OFF: usize = 2 + 2 * MAX_RANK;
pub const BINARY_B_STRIDES_OFF: usize = 3 + 2 * MAX_RANK;
pub const BINARY_B_OFFSET_OFF: usize = 3 + 3 * MAX_RANK;

/// Reduce meta:
/// `[in_rank, reduce_rank, offset, in_shape[MAX_RANK], in_strides[MAX_RANK],
///   axes[MAX_RANK]]`.
pub const REDUCE_META_WORDS: usize = 3 + 3 * MAX_RANK;
pub const REDUCE_SHAPE_OFF: usize = 3;
pub const REDUCE_STRIDES_OFF: usize = 3 + MAX_RANK;
pub const REDUCE_AXES_OFF: usize = 3 + 2 * MAX_RANK;

/// Softmax meta: `[rank, axis, offset, shape[MAX_RANK], strides[MAX_RANK]]`.
pub const SOFTMAX_META_WORDS: usize = 3 + 2 * MAX_RANK;
pub const SOFTMAX_SHAPE_OFF: usize = 3;
pub const SOFTMAX_STRIDES_OFF: usize = 3 + MAX_RANK;

/// Concat meta:
/// `[total, rank, in_shape[MAX_RANK], in_strides[MAX_RANK], in_offset,
///   out_shape[MAX_RANK], axis, axis_start]`.
pub const CONCAT_META_WORDS: usize = 5 + 3 * MAX_RANK;
pub const CONCAT_IN_SHAPE_OFF: usize = 2;
pub const CONCAT_IN_STRIDES_OFF: usize = 2 + MAX_RANK;
pub const CONCAT_IN_OFFSET_OFF: usize = 2 + 2 * MAX_RANK;
pub const CONCAT_OUT_SHAPE_OFF: usize = 3 + 2 * MAX_RANK;
pub const CONCAT_AXIS_OFF: usize = 3 + 3 * MAX_RANK;
pub const CONCAT_AXIS_START_OFF: usize = 4 + 3 * MAX_RANK;

/// Pad meta:
/// `[src_total, rank, src_shape[MAX_RANK], src_strides[MAX_RANK], src_offset,
///   out_shape[MAX_RANK], pads_before[MAX_RANK]]`.
pub const PAD_META_WORDS: usize = 3 + 4 * MAX_RANK;
pub const PAD_SRC_SHAPE_OFF: usize = 2;
pub const PAD_SRC_STRIDES_OFF: usize = 2 + MAX_RANK;
pub const PAD_SRC_OFFSET_OFF: usize = 2 + 2 * MAX_RANK;
pub const PAD_OUT_SHAPE_OFF: usize = 3 + 2 * MAX_RANK;
pub const PAD_PADS_BEFORE_OFF: usize = 3 + 3 * MAX_RANK;

/// Matmul meta:
/// `[batch_rank, M, K, N, a_row_s, a_col_s, b_row_s, b_col_s, a_off, b_off,
///   batch_out_shape[MAX_RANK-2], a_bcast[MAX_RANK-2], b_bcast[MAX_RANK-2]]`.
pub const MATMUL_BATCH_RANK: usize = MAX_RANK - 2;
pub const MATMUL_META_WORDS: usize = 10 + 3 * MATMUL_BATCH_RANK;
pub const MATMUL_BATCH_SHAPE_OFF: usize = 10;
pub const MATMUL_A_BCAST_OFF: usize = 10 + MATMUL_BATCH_RANK;
pub const MATMUL_B_BCAST_OFF: usize = 10 + 2 * MATMUL_BATCH_RANK;
