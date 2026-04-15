# IR Reference

The IR describes a computation graph as pure data. It has no device state, no gradient information, and no execution state. The same IR is produced by `core`'s graph compiler and will eventually be produced by the ONNX parser — both must generate graphs that the runtime can execute without modification.

---

## Types

### `AttributeValue`

```ts
type AttributeValue = number | string | number[] | string[] | boolean | ArrayBuffer | ArrayBufferView;
```

ONNX-style flexible attribute bag. `ArrayBuffer` is used for constant tensor data (weights, biases).

### `Value`

A value is an edge in the graph, not a node. It describes a tensor flowing between ops.

```ts
interface Value {
  name: string;

  shape: (number | null)[];  // null = dynamic dimension
  dtype: 'float32' | 'int32' | 'bool';

  data?: ArrayBuffer;        // only for weights/constants (initializers)

  producer?: string;         // id of the Node that outputs this value
  consumers?: string[];      // ids of Nodes that take this value as input

  debugName?: string;
}
```

`producer` and `consumers` are computed post-hoc by `compileGraph()` after the graph is assembled. They drive reference counting in `Engine` — a tensor is freed when its consumer count drops to zero and it is not in the retained set (outputs, inputs, initializers).

### `Node`

An op instance in the graph. Inputs and outputs are value names (strings), not value objects.

```ts
interface Node {
  id: string;
  op: string;                             // e.g. "Add", "MatMul" — must match registry key exactly
  inputs: string[];                       // Value.name references
  outputs: string[];                      // Value.name references
  attributes?: Record<string, AttributeValue>;
  name?: string;                          // optional human label for visualization
}
```

`op` must use the exact string used as the key in each backend's kernel registry. Convention follows ONNX: PascalCase, e.g. `"MatMul"` not `"matmul"` or `"mat_mul"`.

### `Graph`

```ts
interface Graph {
  nodes: Node[];
  values: Record<string, Value>;  // keyed by Value.name
  inputs: string[];               // value names of graph inputs (trainable tensors)
  outputs: string[];              // value names of graph outputs
  initializers: string[];         // value names of fixed weights/constants (subset of values)
  name?: string;
  opset?: number;                 // ONNX opset version for future compatibility
}
```

`inputs` are tensors that vary per inference call (e.g., the batch `x`). `initializers` are fixed weights that are written once and reused. Both are retained in memory for the full duration of graph execution — only intermediate tensors are freed during execution.

**Note:** `compileGraph()` in `core/src/compiler.ts` currently uses `requiresGrad: true` to classify a tensor as a graph input rather than an initializer. This heuristic works for inference but will need revisiting when the training loop is built — weight tensors have `requiresGrad: true` but should be initializers during inference.

---

## Shape System

```ts
shape: (number | null)[]
```

| Shape | Meaning |
| --- | --- |
| `[1, 128]` | fixed — all dimensions known at compile time |
| `[null, 128]` | dynamic batch — batch size unknown |
| `[null, null]` | fully dynamic |

Dynamic shapes (`null` dimensions) are part of the type system but are not yet enforced by the runtime. The current engine assumes all shapes are static at execution time.

---

## Broadcasting

Broadcasting follows ONNX right-aligned rules:

```
[1, 64]
[64]
→ [1, 64]
```

The utility `broadcastShapes(a, b)` in `packages/core/src/shape.ts` computes the output shape. Backend kernels do not yet implement full broadcasting — the current elementwise kernels only handle same-shape inputs or scalar broadcast (length-1 arrays). See the kernel support matrix in [README.md](../README.md) for current state.

---

## ONNX Alignment

When the ONNX parser is built, it maps directly to this IR:

| ONNX | This IR |
| --- | --- |
| `NodeProto` | `Node` |
| `TensorProto` | `Value` (with `data` field populated) |
| `GraphProto` | `Graph` |

Op names match ONNX exactly (`"MatMul"`, `"Relu"`, `"Softmax"`, etc.) so that a graph produced from ONNX runs on the same kernel registries without any translation layer.

---

## What IR Must Not Contain

- No gradient closures or backward functions (those live on `Tensor._ctx` in `core`)
- No device identifiers or backend references
- No execution state (register maps, intermediate buffers)
- No runtime-specific memory handles

If you find yourself adding any of these to `ir/src/types.ts`, the logic belongs in `core`, `runtime`, or a backend package instead.
