import { Graph, Node } from '@minitensor/ir';
import { Backend, RuntimeTensor, computeContiguousStrides } from './backend';

// ---------------------------------------------------------------------------
// View op computation (pure metadata — no backend allocation or kernel needed)

const VIEW_OPS = new Set(['Transpose', 'Reshape', 'Slice']);

function computeView(node: Node, inputs: RuntimeTensor[]): RuntimeTensor {
  const src = inputs[0];
  switch (node.op) {
    case 'Transpose': {
      // Swap the last two axes of shape and strides.
      const rank = src.shape.length;
      const newShape = [...src.shape];
      const newStrides = [...src.strides];
      newShape[rank - 1] = src.shape[rank - 2];
      newShape[rank - 2] = src.shape[rank - 1];
      newStrides[rank - 1] = src.strides[rank - 2];
      newStrides[rank - 2] = src.strides[rank - 1];
      return {
        storage: src.storage,
        shape: newShape,
        strides: newStrides,
        offset: src.offset,
        dtype: src.dtype,
        isView: true,
      };
    }

    case 'Reshape': {
      const newShape = (node.attributes!.shape as number[]).slice();
      // Total element count must be preserved. Only valid for contiguous tensors
      // (non-contiguous tensors must be copied first with .contiguous()).
      return {
        storage: src.storage,
        shape: newShape,
        strides: computeContiguousStrides(newShape),
        offset: src.offset,
        dtype: src.dtype,
        isView: true,
      };
    }

    case 'Slice': {
      const starts = node.attributes!.starts as number[];
      const ends = node.attributes!.ends as number[];
      const newShape = starts.map((s, i) => ends[i] - s);
      const newOffset = src.offset + starts.reduce((acc, s, i) => acc + s * src.strides[i], 0);
      return {
        storage: src.storage,
        shape: newShape,
        strides: [...src.strides],
        offset: newOffset,
        dtype: src.dtype,
        isView: true,
      };
    }

    default:
      throw new Error(`computeView: unknown view op '${node.op}'`);
  }
}

// ---------------------------------------------------------------------------

export class Engine {
  private backend: Backend;
  private registry = new Map<string, RuntimeTensor>();

  constructor(backend: Backend) {
    this.backend = backend;
  }

  set(
    name: string,
    data: ArrayBufferView,
    shape: (number | null)[],
    dtype: 'float32' | 'int32' | 'bool' = 'float32',
  ) {
    if (this.registry.has(name)) {
      this.backend.dispose(this.registry.get(name)!);
    }
    const tensor = this.backend.allocate(shape, dtype);
    this.backend.write(tensor, data);
    this.registry.set(name, tensor);
  }

  get(name: string): Promise<ArrayBufferView | undefined> {
    const tensor = this.registry.get(name);
    if (!tensor) return Promise.resolve(undefined);
    return this.backend.read(tensor);
  }

  dispose(name: string): void {
    const tensor = this.registry.get(name);
    if (tensor) {
      this.backend.dispose(tensor);
      this.registry.delete(name);
    }
  }

  evaluate(graph: Graph): void {
    const topoNodes = this.topologicalSort(graph);

    const retained = new Set<string>([...graph.outputs, ...graph.inputs, ...graph.initializers]);

    const refCounts = new Map<string, number>();
    for (const key of Object.keys(graph.values)) {
      const consumers = graph.values[key].consumers || [];
      refCounts.set(key, consumers.length);
    }

    // viewParents: view tensor id → source tensor id.
    // When a view is freed (refcount 0), we propagate the decrement to the source
    // so the source's storage is not freed until no view or other consumer needs it.
    const viewParents = new Map<string, string>();

    const decrementRef = (id: string) => {
      const cnt = (refCounts.get(id) || 0) - 1;
      refCounts.set(id, cnt);
      if (cnt === 0 && !retained.has(id)) {
        this.dispose(id);
        // If this was a view, propagate to the source once the view is gone.
        const srcId = viewParents.get(id);
        if (srcId !== undefined) {
          viewParents.delete(id);
          decrementRef(srcId);
        }
      }
    };

    for (const node of topoNodes) {
      if (node.op === 'Constant') {
        const outId = node.outputs[0];
        const outValue = graph.values[outId];
        const rawData = node.attributes?.data as ArrayBufferView;
        if (!rawData) throw new Error(`Constant node ${node.id} missing 'data' buffer attribute`);
        this.set(outId, rawData, outValue.shape, outValue.dtype);
        continue;
      }

      const inputs = node.inputs.map((id) => {
        const t = this.registry.get(id);
        if (!t) throw new Error(`Missing expected tensor input ${id} for node ${node.id}`);
        return t;
      });

      // View ops: create a zero-copy RuntimeTensor view without allocating or executing.
      // The source tensor's refcount is NOT decremented here; instead it is decremented
      // transitively when the view itself is freed (via viewParents).
      if (VIEW_OPS.has(node.op)) {
        const srcId = node.inputs[0];
        const viewId = node.outputs[0];
        this.registry.set(viewId, computeView(node, inputs));
        viewParents.set(viewId, srcId);
        // Do NOT decrement srcId's refcount — we'll do it when the view is freed.
        continue;
      }

      const outputs: RuntimeTensor[] = [];
      for (const outId of node.outputs) {
        const outValue = graph.values[outId];
        if (!outValue) throw new Error(`Missing value definition for output ${outId}`);
        const t = this.backend.allocate(outValue.shape, outValue.dtype);
        this.registry.set(outId, t);
        outputs.push(t);
      }

      this.backend.execute(node, inputs, outputs);

      for (const inId of node.inputs) {
        decrementRef(inId);
      }
    }
  }

  private topologicalSort(graph: Graph): Node[] {
    const result: Node[] = [];
    const visited = new Set<string>();
    const processing = new Set<string>();

    const producerMap = new Map<string, Node>();
    for (const n of graph.nodes) {
      for (const out of n.outputs) {
        producerMap.set(out, n);
      }
    }

    const visit = (node: Node) => {
      if (visited.has(node.id)) return;
      if (processing.has(node.id)) throw new Error(`Cyclic dependency detected at node ${node.id}`);
      processing.add(node.id);
      for (const inId of node.inputs) {
        const producer = producerMap.get(inId);
        if (producer) visit(producer);
      }
      processing.delete(node.id);
      visited.add(node.id);
      result.push(node);
    };

    for (const node of graph.nodes) {
      visit(node);
    }

    return result;
  }
}
