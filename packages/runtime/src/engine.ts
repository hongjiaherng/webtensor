import { Graph, Node, DType, computeContiguousStrides } from '@webtensor/ir';
import { Backend, RuntimeTensor, isContiguous } from './backend';
import { viewRegistry } from './views/registry';

export type BackendFactory = () => Promise<Backend>;

const backendRegistry = new Map<string, BackendFactory>();

export function registerBackend(device: string, factory: BackendFactory): void {
  backendRegistry.set(device, factory);
}

// ---------------------------------------------------------------------------
// View ops: Reshape and View are handled specially (not in the view registry)
// because Reshape may auto-copy and View must verify contiguity.

const RESHAPE_OPS = new Set(['Reshape', 'View']);

// ---------------------------------------------------------------------------

export class Engine {
  private backend: Backend;
  private registry = new Map<string, RuntimeTensor>();
  readonly device?: string;

  constructor(backend: Backend) {
    this.backend = backend;
  }

  static async create(device: string): Promise<Engine> {
    const factory = backendRegistry.get(device);
    if (!factory) throw new Error(`No backend registered for device '${device}'`);
    const engine = new Engine(await factory());
    (engine as { device: string }).device = device;
    return engine;
  }

  set(name: string, data: ArrayBufferView, shape: (number | null)[], dtype: DType = 'float32') {
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

  async evaluate(
    graph: Graph,
    feeds?: Record<string, ArrayBufferView>,
  ): Promise<void> {
    const topoNodes = this.topologicalSort(graph);

    const retained = new Set<string>([...graph.outputs, ...graph.inputs, ...graph.initializers]);

    const refCounts = new Map<string, number>();
    for (const key of Object.keys(graph.values)) {
      const consumers = graph.values[key].consumers || [];
      refCounts.set(key, consumers.length);
    }

    const viewParents = new Map<string, string>();

    const decrementRef = (id: string) => {
      const cnt = (refCounts.get(id) || 0) - 1;
      refCounts.set(id, cnt);
      if (cnt === 0 && !retained.has(id)) {
        this.dispose(id);
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

      if (node.op === 'Placeholder') {
        const outId = node.outputs[0];
        const outValue = graph.values[outId];
        // Use the explicit feed if supplied; otherwise fall back to embedded
        // default data (set by `tensor(..., { requiresGrad: true })` and the
        // other `requiresGrad: true` factories). This lets the same tensor
        // work eagerly (`evaluate(graph)`) and through `compile(...)`.
        const feedData =
          feeds?.[outId] ?? (node.attributes?.data as ArrayBufferView | undefined);
        if (!feedData) {
          throw new Error(
            `Missing feed for placeholder '${outId}' — pass via evaluate(graph, feeds) ` +
              'or create the tensor with `{ requiresGrad: true }` so its data is embedded.',
          );
        }
        this.set(outId, feedData, outValue.shape, outValue.dtype);
        continue;
      }

      const inputs = node.inputs.map((id) => {
        const t = this.registry.get(id);
        if (!t) throw new Error(`Missing expected tensor input ${id} for node ${node.id}`);
        return t;
      });

      // View registry dispatch (zero-copy metadata ops)
      const viewFn = viewRegistry.get(node.op);
      if (viewFn) {
        const srcId = node.inputs[0];
        const viewId = node.outputs[0];
        this.registry.set(viewId, viewFn(node, inputs[0]));
        viewParents.set(viewId, srcId);
        continue;
      }

      // Reshape / View — special handling for contiguity
      if (RESHAPE_OPS.has(node.op)) {
        const src = inputs[0];
        const srcId = node.inputs[0];
        const outId = node.outputs[0];
        const newShape = (node.attributes!.shape as number[]).slice();

        if (node.op === 'View') {
          // View is strict: throw if non-contiguous
          if (!isContiguous(src.shape as number[], src.strides, src.offset)) {
            throw new Error('view() requires a contiguous tensor; call .contiguous() first');
          }
          this.registry.set(outId, {
            storage: src.storage,
            shape: newShape,
            strides: computeContiguousStrides(newShape),
            offset: src.offset,
            dtype: src.dtype,
            isView: true,
          });
          viewParents.set(outId, srcId);
          continue;
        }

        // Reshape: auto-copy if non-contiguous (PyTorch semantics)
        if (isContiguous(src.shape as number[], src.strides, src.offset)) {
          this.registry.set(outId, {
            storage: src.storage,
            shape: newShape,
            strides: computeContiguousStrides(newShape),
            offset: src.offset,
            dtype: src.dtype,
            isView: true,
          });
          viewParents.set(outId, srcId);
        } else {
          // Non-contiguous: execute a Contiguous copy then reshape
          const contiguousTensor = this.backend.allocate(src.shape, src.dtype);
          this.backend.execute(
            {
              id: `${node.id}_auto_contig`,
              op: 'Contiguous',
              inputs: [],
              outputs: [],
              name: 'auto_contiguous',
            },
            [src],
            [contiguousTensor],
          );
          this.registry.set(outId, {
            storage: contiguousTensor.storage,
            shape: newShape,
            strides: computeContiguousStrides(newShape),
            offset: 0,
            dtype: src.dtype,
            isView: false,
          });
          for (const inId of node.inputs) {
            decrementRef(inId);
          }
        }
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

      await this.backend.execute(node, inputs, outputs);

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
