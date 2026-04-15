import { Tensor } from './tensor';
import { Graph, Node, Value } from '@minitensor/ir';

export function compileGraph(outputs: Tensor[]): Graph {
  const nodes: Node[] = [];
  const values: Record<string, Value> = {};
  
  const visitedTensors = new Set<string>();
  const visitedOps = new Set<string>();
  const inputs: string[] = [];
  const initializers: string[] = [];

  const traverse = (t: Tensor) => {
    if (visitedTensors.has(t.id)) return;
    visitedTensors.add(t.id);

    values[t.id] = {
      name: t.id,
      shape: t.shape,
      dtype: t.dtype,
    };

    if (t._ctx) {
      for (const input of t._ctx.inputs) {
        if (input instanceof Tensor) {
          traverse(input);
        }
      }

      const nodeId = `node_${t.id}`;
      if (!visitedOps.has(nodeId)) {
        visitedOps.add(nodeId);
        
        nodes.push({
          id: nodeId,
          op: t._ctx.op,
          inputs: t._ctx.inputs.map((inT) => (inT as Tensor).id),
          outputs: [t.id],
          attributes: t._ctx.attributes,
        });

        values[t.id].producer = nodeId;
      }
      
      if (t._ctx.op === 'Constant') {
        if (t.requiresGrad) {
          inputs.push(t.id);
        } else {
          initializers.push(t.id);
        }
      }
    } else {
      // Leaf tensors
      if (t.requiresGrad) {
        inputs.push(t.id);
      } else {
        // It's a constant or weight initializer
        initializers.push(t.id);
      }
    }
  };

  for (const out of outputs) {
    traverse(out);
  }

  // Optional: compute consumers for Values
  for (const node of nodes) {
    for (const inputId of node.inputs) {
      const val = values[inputId];
      if (val) {
        if (!val.consumers) val.consumers = [];
        val.consumers.push(node.id);
      }
    }
  }

  return {
    nodes,
    values,
    inputs,
    outputs: outputs.map((out) => out.id),
    initializers,
  };
}
