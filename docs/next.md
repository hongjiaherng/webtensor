# Roadmap Checklist

Check an item off (`[x]`) when it is fully implemented and tested on all target backends.

---

## Priority: Next Three

### 1. Batched MatMul (rank ≥ 3, max 64)

**What it is:** Extend MatMul from strictly 2D (`[M,K] × [K,N]`) to arbitrary batch dimensions (`[..., M, K] × [..., K, N] → [..., M, N]`), with a hard rank ceiling of 64 matching PyTorch's `MAX_TENSORIMPL_DIMS`.

**Why it matters:** Real models — transformers, CNNs with batched projections — use batched matmul constantly. Without it, webtensor can only prototype toy 2D problems.

**What needs to change:**

- **Max-rank constraint** — add a guard at `allocate()` in all three backends throwing if `shape.length > 64`. One line each, do this first so the constraint is enforced everywhere from day one.
- **CPU kernel** — loop over all batch dimensions (iterate the flat batch index, compute batch offsets via strides), then call the existing 2D inner loop. Strides already support arbitrary layout, so no storage changes are needed.
- **WASM kernel** — same logic in Rust. The `matmul_raw` function signature needs batch dimension parameters or a stride-based interface matching the CPU approach.
- **WebGPU kernel** — the current shader hardcodes 2D. Replace the `@compute` entry point with a 3D dispatch: `x=rows, y=cols, z=batch_flat`. The batch flat index is decomposed into per-axis coordinates using the existing `TensorMeta` strides for all dims except the last two. The last two dims remain the `M/K/N` contraction. The `TensorMeta` uniform (8 shape + 8 stride slots) already supports up to rank 8; for rank > 8 a different packing strategy is needed — either increase the slot count or restrict batched matmul to ≤ 8 total dims as a documented limitation.
- **Core `ops.ts`** — update the shape-inference logic in `matmul()` to handle batch dims and validate that the last two dims of A and B are compatible.
- **Tests** — add `[2,2,3] × [2,3,4] → [2,2,4]` and `[3,1,4,2] × [3,1,2,5] → [3,1,4,5]` cases to `tests/ops/matmul.test.ts`.

---

### 2. Reduce Ops (Sum / Mean)

**What it is:** Dimension-reducing operations — `sum(a, axes?, keepdim?)` and `mean(a, axes?, keepdim?)`.

**Why it matters:** Reductions are critical for loss functions (MSE, CrossEntropy), gradient unbroadcasting in autograd, and normalization layers. Without them, training is impossible.

**What needs to change:**

- **Core op** in `ops.ts` — define `sum()` and `mean()`. `mean` can be composed from `sum + div`.
- **CPU kernel** — loop over output elements, accumulate over reduced dimensions using strided indexing.
- **WASM kernel** — same logic in Rust.
- **WebGPU kernel** — each thread computes one output element by accumulating over the reduction space.
- **Backward** — `sum` backward: `expand(grad, input.shape)`. `mean` backward: same, divided by count.

---

### 3. Dtype Kernels

**What it is:** Extend op kernels beyond `float32` to support `int32` (and eventually `bool`). The dtype infrastructure (type system, `allocate()`, `read()`, `write()`, `TypedArray` union) is already in place — what's missing is per-dtype kernel support.

**Why it matters:** `int32` is needed for index ops (Gather, Scatter, embedding lookups). `bool` is needed for masks and logical ops.

**Current state:** All three dtypes (`float32`, `int32`, `bool`) can be allocated and round-tripped. Only `float32` has op kernels.

---

## Ops

| Op                        | CPU | WASM | WebGPU |
| ------------------------- | :-: | :--: | :----: |
| Add                       | [x] | [x]  |  [x]   |
| Sub                       | [x] | [x]  |  [x]   |
| Mul                       | [x] | [x]  |  [x]   |
| Div                       | [x] | [x]  |  [x]   |
| MatMul (2D)               | [x] | [x]  |  [x]   |
| MatMul (rank ≥ 3, max 64) | [ ] | [ ]  |  [ ]   |
| Transpose                 | [x] | [x]  |  [x]   |
| Relu                      | [x] | [x]  |  [x]   |
| Neg                       | [x] | [x]  |  [x]   |
| Exp                       | [x] | [x]  |  [x]   |
| Log                       | [x] | [x]  |  [x]   |
| Sqrt                      | [x] | [x]  |  [x]   |
| Abs                       | [x] | [x]  |  [x]   |
| Pow                       | [x] | [x]  |  [x]   |
| Sigmoid                   | [x] | [x]  |  [x]   |
| Tanh                      | [x] | [x]  |  [x]   |
| Softmax                   | [ ] | [ ]  |  [ ]   |
| Reshape                   | [x] | [x]  |  [x]   |
| Slice                     | [x] | [x]  |  [x]   |
| Concat                    | [ ] | [ ]  |  [ ]   |
| Reduce (Sum / Mean)       | [ ] | [ ]  |  [ ]   |

## Autograd

| Feature                    | Done |
| -------------------------- | :--: |
| Add backward               | [x]  |
| Mul backward               | [x]  |
| MatMul backward            | [x]  |
| Transpose backward         | [x]  |
| Relu backward — CPU + WASM | [x]  |
| Relu backward — WebGPU     | [x]  |
| Sub backward               | [x]  |
| Div backward               | [x]  |
| Reduce backward            | [ ]  |
| Sigmoid / Tanh backward    | [x]  |
| Softmax backward           | [ ]  |

## Training

| Feature                 | Done |
| ----------------------- | :--: |
| Loss: MSE               | [ ]  |
| Loss: CrossEntropy      | [ ]  |
| Optimizer: SGD          | [ ]  |
| Optimizer: Adam         | [ ]  |
| Training loop in `core` | [ ]  |

## Dtypes

A cell is checked when the dtype is fully supported on that backend: type system, `allocate()`, and kernels for all applicable ops.

| Dtype     | CPU | WASM | WebGPU | Notes                                           |
| --------- | :-: | :--: | :----: | ----------------------------------------------- |
| `float32` | [x] | [x]  |  [x]   | Primary dtype — full kernel support             |
| `int32`   | [ ] | [ ]  |  [ ]   | Allocate + round-trip works; op kernels missing |
| `bool`    | [ ] | [ ]  |  [ ]   | Allocate + round-trip works; no op kernels      |

## Infrastructure

| Feature                                         | Done |
| ----------------------------------------------- | :--: |
| Strided tensor model (strides, offset, views)   | [x]  |
| Broadcasting via stride-0                       | [x]  |
| Cross-backend parity tests                      | [x]  |
| Max rank = 64 (package-wide constraint)         | [ ]  |
| Single async `evaluate()` (awaits all backends) | [x]  |
| Device dispatch via `Engine.create(device)`     | [x]  |
| DType infrastructure (TypedArray, utilities)    | [x]  |

## Packages

| Package                                             | Done |
| --------------------------------------------------- | :--: |
| `packages/onnx` — protobuf parser, op mapping to IR | [ ]  |
| `packages/devtools` — graph viz, training dashboard | [ ]  |
