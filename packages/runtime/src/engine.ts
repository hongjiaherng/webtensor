import { Graph, Node } from '@minitensor/ir';
import { Backend, RuntimeTensor } from './backend';

export class Engine {
  private backend: Backend;
  private registry = new Map<string, RuntimeTensor>();

  constructor(backend: Backend) {
    this.backend = backend;
  }

  set(name: string, data: ArrayBufferView, shape: (number | null)[], dtype: 'float32' | 'int32' | 'bool' = 'float32') {
    // If it already exists, overwrite it or dispose old one
    if (this.registry.has(name)) {
      this.backend.dispose(this.registry.get(name)!);
    }
    const tensor = this.backend.allocate(shape, dtype);
    this.backend.write(tensor, data);
    this.registry.set(name, tensor);
  }

  get(name: string): ArrayBufferView | Promise<ArrayBufferView> | undefined {
    const tensor = this.registry.get(name);
    if (!tensor) return undefined;
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
    
    // Build isolated JIT Sweeper mechanics using Native Graph tracking
    const retained = new Set<string>([
      ...graph.outputs, 
      ...graph.inputs, 
      ...graph.initializers
    ]);
    
    const refCounts = new Map<string, number>();
    for (const key of Object.keys(graph.values)) {
      const consumers = graph.values[key].consumers || [];
      refCounts.set(key, consumers.length);
    }

    for (const node of topoNodes) {
      if (node.op === 'Constant') {
        const outId = node.outputs[0];
        const outValue = graph.values[outId];
        const rawData = node.attributes?.data as ArrayBufferView;
        if (!rawData) throw new Error(`Constant node ${node.id} missing 'data' buffer attribute`);
        this.set(outId, rawData, outValue.shape, outValue.dtype);
        continue;
      }

      const inputs = node.inputs.map(id => {
        const t = this.registry.get(id);
        if (!t) throw new Error(`Missing expected tensor input ${id} for node ${node.id}`);
        return t;
      });

      const outputs: RuntimeTensor[] = [];
      for (const outId of node.outputs) {
        const outValue = graph.values[outId];
        if (!outValue) throw new Error(`Missing value definition for output ${outId}`);
        
        // Allocate buffer for this output
        const t = this.backend.allocate(outValue.shape, outValue.dtype);
        this.registry.set(outId, t);
        outputs.push(t);
      }

      this.backend.execute(node, inputs, outputs);

      // Perform Automated Reference Freeing
      for (const inId of node.inputs) {
        const currentCount = (refCounts.get(inId) || 0) - 1;
        refCounts.set(inId, currentCount);
        
        // Exclusively drop buffers natively mapped into physical GPU structures!
        if (currentCount === 0 && !retained.has(inId)) {
          this.dispose(inId);
        }
      }
    }
  }

  private topologicalSort(graph: Graph): Node[] {
    const result: Node[] = [];
    const visited = new Set<string>();
    const processing = new Set<string>();
    
    // Map nodes by their outputs for quick dependency lookup
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
        // If it's undefined, it means `inId` is an external Graph input, so skip it.
        if (producer) {
          visit(producer);
        }
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
