# WebGPU Option A: Native Strided Ops

This document captures the design and the work-in-progress state of the native-strided
WebGPU kernel implementation (Option A), so it can be resumed later.

---

## Background: Option A vs Option B

**Option B (current state)** — make every input contiguous before the kernel runs.

- A gather pre-pass shader copies non-contiguous tensors into packed GPU buffers.
- Kernels can be simple: they always receive flat, contiguous data.
- Extra GPU dispatches and allocations for every non-contiguous input.
- Lives in `WebGPUBackend.gatherPass()` + `kernels/internal/gather.wgsl`.

**Option A (target state)** — kernels handle arbitrary strides natively.

- No gather pre-pass; kernels receive the raw GPUBuffer plus a meta storage buffer
  that encodes the shape, strides, and offset for each input.
- Kernels perform the same strided-index decomposition as the CPU backend.
- Fewer GPU allocations and dispatches; correct zero-copy view semantics.

---

## What was implemented

All five kernels had their WGSL shaders rewritten and their TS wrapper files updated:

| Kernel    | WGSL                    | TS wrapper            | Meta layout                  |
| --------- | ----------------------- | --------------------- | ---------------------------- |
| Add       | `elementwise/add.wgsl`  | `elementwise/add.ts`  | `buildBinaryMeta` (28 u32)   |
| Mul       | `elementwise/mul.wgsl`  | `elementwise/mul.ts`  | `buildBinaryMeta` (28 u32)   |
| Relu      | `elementwise/relu.wgsl` | `elementwise/relu.ts` | `buildUnaryMeta` (19 u32)    |
| MatMul    | `linear/matmul.wgsl`    | `linear/matmul.ts`    | `buildMatmulMeta` (9 u32)    |
| Transpose | `shape/transpose.wgsl`  | `shape/transpose.ts`  | `buildTransposeMeta` (5 u32) |

The helper functions `buildBinaryMeta`, `buildUnaryMeta`, `buildMatmulMeta`, and
`buildTransposeMeta` were added to `kernels/utils.ts` and remain there even in the
current Option B state; they are unused but preserved for Option A.

`WebGPUBackend.execute()` was updated to remove the gather pre-pass and pass inputs
directly to kernels. This was then **reverted** when tests showed all-zero output.

---

## Meta buffer layouts

All meta buffers are `var<storage, read> meta: array<u32>` (not `var<uniform>`) to
avoid WGSL's 16-byte element-padding rule that applies to `array<u32, N>` inside
uniform structs. They are created with `GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST`
and filled via `mappedAtCreation: true` + `unmap()`.

### Binary elementwise (add, mul) — `buildBinaryMeta` — 28 u32 = 112 bytes

```
[0]       total elements in output
[1]       rank
[2..9]    out_shape[0..7]      (padded with 0 for dims < 8)
[10..17]  a_broadcast_strides[0..7]
[18]      a_offset
[19..26]  b_broadcast_strides[0..7]
[27]      b_offset
```

Broadcast strides come from `broadcastStridesOf(outShape, inShape, inStrides)`:
broadcast axes (size-1 in input) get stride 0 so the same element is read repeatedly.

### Unary elementwise (relu) — `buildUnaryMeta` — 19 u32 = 76 bytes (padded to 80)

```
[0]       total elements
[1]       rank
[2..9]    shape[0..7]
[10..17]  strides[0..7]
[18]      offset
```

### MatMul — `buildMatmulMeta` — 9 u32 = 36 bytes (padded to 48)

```
[0]  M
[1]  K
[2]  N
[3]  a_row_stride   (A.strides[rank-2], or K for rank-1)
[4]  a_col_stride   (A.strides[rank-1])
[5]  b_row_stride
[6]  b_col_stride
[7]  a_offset
[8]  b_offset
```

### Transpose — `buildTransposeMeta` — 5 u32 = 20 bytes (padded to 32)

```
[0]  M            (rows of input)
[1]  N            (cols of input)
[2]  row_stride   (input.strides[rank-2])
[3]  col_stride   (input.strides[rank-1])
[4]  offset
```

---

## WGSL strided index decomposition (binary case)

```wgsl
var rem   = i;            // flat output index
var a_idx = meta[18];     // a_offset
var b_idx = meta[27];     // b_offset

for (var d = 0u; d < 8u; d = d + 1u) {
    let axis  = rank - 1u - d;      // innermost first
    if (d >= rank) { break; }
    let dim   = meta[2u + axis];
    let coord = rem % dim;
    rem       = rem / dim;
    a_idx    += coord * meta[10u + axis];
    b_idx    += coord * meta[19u + axis];
}

out[i] = a[a_idx] + b[b_idx];
```

Note: `axis = rank - 1u - d` overflows (wraps to u32::MAX) when `d == rank`, but the
`break` fires before `axis` is used for any memory access, so this is safe.

---

## Bind group layout changes (per kernel)

### Binary (add, mul) — 4 bindings

```
binding 0: var<storage, read>       a    : array<f32>   (input A)
binding 1: var<storage, read>       b    : array<f32>   (input B)
binding 2: var<storage, read_write> out  : array<f32>   (output)
binding 3: var<storage, read>       meta : array<u32>   (meta buffer)
```

### Unary (relu) — 3 bindings

```
binding 0: var<storage, read>       a    : array<f32>
binding 1: var<storage, read_write> out  : array<f32>
binding 2: var<storage, read>       meta : array<u32>
```

### MatMul — 4 bindings (meta replaces uniform)

```
binding 0: var<storage, read>       A    : array<f32>
binding 1: var<storage, read>       B    : array<f32>
binding 2: var<storage, read_write> Out  : array<f32>
binding 3: var<storage, read>       meta : array<u32>   (replaces var<uniform>)
```

### Transpose — 3 bindings (meta replaces uniform)

```
binding 0: var<storage, read>       input  : array<f32>
binding 1: var<storage, read_write> output : array<f32>
binding 2: var<storage, read>       meta   : array<u32>   (replaces var<uniform>)
```

---

## Why it was reverted

After implementing all five shaders and removing the gather pre-pass, all 34
WebGPU-specific tests returned all-zero output. The failure pattern — every thread
exits at `if (i >= total) { return; }` — strongly suggests `meta[0]` (total elements)
is being read as 0.

Possible root causes (uninvestigated):

1. **Silent WGSL shader compilation failure** — if the new WGSL has a validation
   error, `createComputePipeline` produces an invalid pipeline that silently dispatches
   nothing. The original code does not call `shaderModule.getCompilationInfo()`.
   **Recommended first debug step**: add compilation info checks.

2. **Meta buffer lifetime** — the meta buffer is created inside `buildBindGroupEntries`
   and held only by the returned `GPUBindGroupEntry`. If V8 GC collects the bind group
   before the GPU reads the buffer, the buffer is destroyed. However, this should not
   happen per the WebGPU spec: submitted work retains resources until GPU completion.

3. **Storage buffer read access** — `var<storage, read>` requires the bind group to
   declare the buffer with read access. With `layout: 'auto'` WebGPU infers this from
   the shader, so it should be automatic.

4. **Binding index mismatch** — the pipeline layout for old kernels had 3 bindings;
   new kernels have 4. If the pipeline cache accidentally returns an old pipeline built
   before the shader was updated, the bind group layout would not match. The cache key
   is just the op name string, so a stale cached pipeline would cause this.
   **Recommended second debug step**: clear the pipeline cache or add a version key.

---

## Recommended debugging approach for resuming

```typescript
// In each kernel's createPipeline:
const shaderModule = device.createShaderModule({ code: source, label: 'AddShader' });
shaderModule.getCompilationInfo().then((info) => {
  if (info.messages.some((m) => m.type === 'error')) {
    console.error('WGSL compile error:', info.messages);
  }
});
```

Also add a GPU error scope around pipeline creation:

```typescript
device.pushErrorScope('validation');
const pipeline = device.createComputePipeline({ ... });
device.popErrorScope().then(err => {
  if (err) console.error('Pipeline error:', err);
});
```

---

## Files to change when resuming Option A

| File                                | Change needed                                               |
| ----------------------------------- | ----------------------------------------------------------- |
| `src/kernels/elementwise/add.wgsl`  | Replace with Option A shader (see WGSL above)               |
| `src/kernels/elementwise/mul.wgsl`  | Same, with `*` instead of `+`                               |
| `src/kernels/elementwise/relu.wgsl` | Replace with unary strided shader                           |
| `src/kernels/linear/matmul.wgsl`    | Replace with strided matmul shader                          |
| `src/kernels/shape/transpose.wgsl`  | Replace with strided transpose shader                       |
| `src/kernels/elementwise/add.ts`    | Use `buildBinaryMeta`, 4-entry bind group                   |
| `src/kernels/elementwise/mul.ts`    | Same                                                        |
| `src/kernels/elementwise/relu.ts`   | Use `buildUnaryMeta`, 3-entry bind group                    |
| `src/kernels/linear/matmul.ts`      | Use `buildMatmulMeta`, 4-entry bind group                   |
| `src/kernels/shape/transpose.ts`    | Use `buildTransposeMeta`, 3-entry bind group                |
| `src/backend.ts`                    | Remove `gatherPass`/`getGatherPipeline`; simplify `execute` |
| `src/kernels/internal/gather.wgsl`  | Can be deleted once Option A is verified                    |

The helper functions (`buildBinaryMeta`, `buildUnaryMeta`, `buildMatmulMeta`,
`buildTransposeMeta`) already exist in `src/kernels/utils.ts` — no changes needed there.
