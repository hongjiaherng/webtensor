import { Tensor } from './tensor';
import { Graph, Node, Value } from '@webtensor/ir';

/**
 * Compiles a set of output tensors into a pure IR Graph by tracing the
 * computation graph backwards from each output.
 *
 * Classification rules:
 *   initializers — Constant nodes: data is embedded in the graph and owned by
 *                  the graph definition (weights, biases, fixed inputs).
 *                  The engine retains these for the full evaluation lifetime.
 *
 *   inputs       — Reserved for future placeholder tensors: values whose data is
 *                  supplied by the caller at each evaluation (e.g. batch images).
 *                  Not yet used; populated when Tensor.placeholder() is added.
 *
 * IMPORTANT: requiresGrad is an autograd concern and has NO effect on whether a
 * tensor is an initializer or a runtime input. A weight tensor is requiresGrad:true
 * AND an initializer. A batch input tensor is requiresGrad:false AND a graph input.
 * Conflating the two will break training — classify only on data provenance, never
 * on the gradient flag.
 */
export function compileGraph(outputs: Tensor[]): Graph {
  const nodes: Node[] = [];
  const values: Record<string, Value> = {};

  const visitedTensors = new Set<string>();
  const visitedOps = new Set<string>();
  const inputs: string[] = []; // reserved — see note above
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
        // Constant nodes always go to initializers — data is embedded in the graph.
        // requiresGrad does NOT affect this classification.
        initializers.push(t.id);
      }
    } else {
      // Leaf tensor with no producer op.  In normal usage this shouldn't happen
      // (tensor() always produces a Constant node), but treat it as an initializer
      // so it is retained during evaluation.
      initializers.push(t.id);
    }
  };

  for (const out of outputs) {
    traverse(out);
  }

  // Compute consumers for Values (used by the engine for reference-count GC).
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
