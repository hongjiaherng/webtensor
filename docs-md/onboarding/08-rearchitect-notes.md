# 08 — Rearchitect Notes

Forward-looking. For each significant future change, what it touches across the codebase, what it unblocks, and what to be careful about. Not a roadmap — see [next.md](../next.md) for that. This is the **impact analysis**.

---

## 1. Reductions (sum / mean / max / argmax)

**Why first:** unblocks gradient unbroadcasting, loss functions, normalization layers, accuracy metrics. Highest-leverage single change in the codebase.

**Touches:**

- `packages/core/src/ops.ts` — add `sum(a, axes?, keepDims?)` and `mean(...)` op functions with backward closures (broadcast back the gradient).
- `packages/core/src/shape.ts` — shape inference for axis-reductions.
- All three backends — new `Sum` (and `Mean` if not done as `Sum / N`) kernel:
  - CPU: straightforward — outer loop over output, inner loop accumulating along reduce axes.
  - WASM: same as CPU, in Rust.
  - WebGPU: tree reduction inside a workgroup with `workgroupBarrier()`; partial outputs combined in a second pass for large reductions.
- `packages/backend-*/src/kernels/registry.ts` — register new kernels.
- Tests — `tests/ops/sum.test.ts`, parity tests, gradient tests.

**Then immediately:** apply unbroadcast in `add` / `sub` / `mul` / `div` backward. Implement `Expand` backward (sum over expanded axes).

**Risk:** the WebGPU implementation needs care — naive serial accumulation in a single thread is slow; a proper hierarchical reduction is the right pattern.

---

## 2. Batched MatMul (rank ≥ 3)

**Why:** every transformer, every batched MLP, every conv-via-im2col needs this.

**Touches:**

- `packages/core/src/ops.ts:97-134` — fix shape inference (the placeholder comment at line 110 is the entry point). Output shape = `broadcast(a.shape[:-2], b.shape[:-2]) + [a.shape[-2], b.shape[-1]]`.
- CPU matmul kernel — extra outer loop over batch indices.
- WASM matmul kernel — same in Rust.
- WebGPU matmul shader — 3D dispatch; batch index in `gid.z`.
- **Prerequisite:** WebGPU `TensorMeta` redesign (next entry) before any batched op pushes rank above 8.
- Backward — current `(grad @ Bᵀ, Aᵀ @ grad)` works if `transpose` and `matmul` both handle batch dims. Verify gradients of broadcast-batched matmul.

**Risk:** broadcast semantics for the leading batch dims (PyTorch-style). Don't forget to broadcast in the backward as well.

---

## 3. WebGPU `TensorMeta` redesign for rank > 8

**Why:** unblocks rank > 8 ops on WebGPU. Eliminates a class of silent corruption bugs.

**Current:** `TensorMeta` is an 80-byte uniform with `array<vec4<u32>, 2>` for shape and strides — exactly 8 slots each. Defined at [packages/backend-webgpu/src/kernels/utils.ts:8-29](../../packages/backend-webgpu/src/kernels/utils.ts#L8-L29).

**Options:**

- **A. Storage buffer instead of uniform.** Replace the uniform with a `var<storage, read>` buffer of arbitrary length. Slightly slower binding lookup; no rank limit.
- **B. Two-uniform split.** Keep uniform-buffer speed for rank ≤ 8 fast path; emit a different shader variant for rank > 8 that uses storage. Higher complexity.
- **C. Ragged packing.** Encode shape and strides into a single dense `array<u32>` with a length prefix. Requires WGSL indexing math.

**Recommended:** A. Simple and correct. If profiling later shows uniform-buffer reads are meaningfully faster, add the fast-path variant.

**Touches:**

- `packages/backend-webgpu/src/kernels/utils.ts` — `packMeta`, `createMetaBuffer`, the `TensorMeta` doc comment.
- Every `.wgsl` shader — change `var<uniform> u_meta: TensorMeta` to `var<storage, read> u_meta: array<u32>` and adjust indexing.
- A migration test that constructs a rank-9 tensor and verifies correctness.

**Quick win first:** add a runtime guard that throws when `rank > 8` until the redesign is shipped.

---

## 4. Int32 / Bool op kernels

**Why:** unblocks embedding lookups (gather with int32 indices), boolean masking, integer arithmetic.

**Touches:**

- All three backends — for each kernel, branch on `inputs[0].dtype` and dispatch to a typed implementation. Most arithmetic ops naturally extend; comparison ops (`Eq`, `Lt`, etc.) produce `bool` outputs.
- `packages/runtime/src/dtype.ts` already has `bytesPerElement` and `typedArrayCtor` for all three dtypes — no infra change needed.
- `packages/core/src/ops.ts` — type-aware result dtype inference (e.g. `add(int32, int32) → int32`; `add(int32, float32) → float32` per a promotion rule TBD).
- WASM — Rust generics or per-dtype duplicated functions; the meta buffer doesn't carry dtype today, so add it or use distinct exports.
- WebGPU — separate WGSL shaders per dtype (WGSL doesn't have generics).

**Risk:** dtype-promotion rules are a design choice that will need to be locked in (PyTorch and NumPy have subtly different rules). Pick PyTorch's and document.

---

## 5. Loss functions and optimizers

**Blocked on reductions** (need `mean`, `sum`).

**Touches (new code, mostly composable from existing ops):**

- A new module — possibly `packages/core/src/nn/loss.ts` and `nn/optim.ts`.
- `mse_loss(pred, target)` = `mean(pow(sub(pred, target), 2))`.
- `cross_entropy(logits, target)` requires `log_softmax` first — that requires reductions plus a numerically stable implementation.
- Optimizers — need to either expose in-place ops (`add_`, `mul_`) or have the user replace the weight tensor each step. The simpler short-term path is the latter (functional update); in-place ops require more engine changes.

**Open question:** do weights live as `Tensor` (in core) or as a separate `Parameter` wrapper? The latter mirrors PyTorch and gives a clean place to attach optimizer state.

---

## 6. ONNX import

**Why:** load pretrained models, prove the IR design.

**Touches:**

- New package `packages/onnx` with a single function `parseOnnx(buffer: ArrayBuffer): Graph`.
- Reads the protobuf, walks the model's nodes, emits webtensor IR (`Node` per ONNX op, `Value` per tensor, embedded weights as `Constant` initializers).
- Op coverage is the long pole — start with a small allowlist (Add, Mul, MatMul, Relu, Softmax, Conv, BatchNorm, ...).
- Likely needs `softmax`, `concat`, `conv2d`, `batchnorm` kernels first — all currently missing.
- A `tests/onnx/` folder with small reference models from the ONNX zoo.

**Risk:** ONNX has hundreds of ops; pick a tight initial scope (e.g. enough for ResNet-18 inference) and grow from there.

---

## 7. Distribution hardening

**Why:** today's `dist/` artifacts are validated only via workspace aliases. Until something imports `@webtensor/core` from a real `node_modules`, publish is a leap of faith.

**Touches:**

- Align package versions (`backend-wasm` is 0.1.0, others 0.0.0). Either bump everything or set up `changesets`.
- Add `license`, `repository`, `author` fields to each `package.json`.
- Add a CI job (GitHub Actions): `bun install`, build WASM, `bun run lint`, `bun run format:check`, `bun run test` in headless browser.
- Add an `examples/` dir with one tiny app (XOR classifier or matmul benchmark) that imports from the published packages — this is the only way to catch packaging regressions.
- Mark `@webgpu/types` as a peer dep of `backend-webgpu` (currently a dev dep at the workspace root).

**Risk:** the `publish:all` script uses `bun publish` which requires Bun ≥ 1.2.0 with the built-in publish command. Validate before relying on it.

---

## 8. Possible larger refactors (no immediate driver)

These aren't needed today, but worth noting if you ever hit them.

### Split eager API from compile API

Today the `Tensor` class entangles eager construction with autograd state. If you wanted a "graph-only" API (build IR directly, no eager Tensors), you'd refactor `_ctx` and `backward()` out of `Tensor` into a separate `GraphBuilder`. Useful if you ever want to support multiple frontends (e.g. a Python-flavored DSL transpiled to JS).

### Shape-inference pass

`compileGraph` doesn't validate shapes — it just takes whatever the eager ops set. A separate `inferShapes(graph)` pass that re-derives every value's shape from inputs + op semantics would catch op-level shape bugs (e.g. matmul mismatch) at compile time instead of runtime, and would help a future ONNX importer that has thinner shape info.

### Engine becomes pluggable per-op

Today the engine has hardcoded fast paths for view ops and reshape. If ops grow (e.g. fusing matmul + bias + activation), it might be cleaner to let the engine consult an extensible "executor strategy" map per op name — `{ View, Reshape, Constant, Default(kernel) }` are first-class, plus the registry is open for fusion strategies.

### Memory pool / tensor reuse

CPU/WASM allocate fresh typed arrays every kernel invocation; WebGPU allocates fresh `GPUBuffer` per output. A simple size-bucketed pool (free-list keyed on `dtype × byteLength`) would dramatically reduce allocator pressure in training loops. Touches `Backend.allocate()` and `Backend.dispose()` only — no API change.

### Async kernels (real parallelism)

WASM with threads (atomics + shared memory) and WebGPU with multiple queue submissions could parallelize independent subgraphs. Today the engine processes nodes strictly serially. A topological scheduler that submits all ready nodes at once — and only awaits when a dependency is needed for the next batch — would help on small ops. Probably not worth it until the workload demands it.

---

## Summary table — order of attack

| #   | Change                                           | Unblocks                                            | Estimated scope       |
| --- | ------------------------------------------------ | --------------------------------------------------- | --------------------- |
| 1   | Reductions (sum/mean)                            | Unbroadcast in autograd; loss functions; expand bwd | 2-3 days × 3 backends |
| 2   | Apply unbroadcast in autograd                    | Correct gradients with broadcasting                 | 0.5 day               |
| 3   | Implement missing backwards (Expand, Slice, Abs) | Full coverage of existing ops                       | 1-2 days              |
| 4   | WebGPU rank > 8 guard                            | Eliminates silent corruption                        | 1 hour                |
| 5   | WebGPU `TensorMeta` redesign                     | Batched matmul on WebGPU                            | 1-2 days              |
| 6   | Batched matmul                                   | Real models (transformers, batched MLPs)            | 3-4 days              |
| 7   | Loss + optimizer                                 | Actual training loop                                | 2-3 days              |
| 8   | Int32 / bool kernels                             | Embedding lookups, masking                          | 3-5 days × 3 backends |
| 9   | Distribution hardening + CI                      | Confidence in publish                               | 1-2 days              |
| 10  | ONNX import                                      | Pretrained models                                   | weeks                 |

For overall direction see [next.md](../next.md). For every concrete bug see [06-bugs-and-gaps.md](06-bugs-and-gaps.md).
