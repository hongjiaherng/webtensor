import { Tensor } from '../tensor';
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
 * @category Compile
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
      } else if (t._ctx.op === 'Placeholder') {
        // Placeholder nodes are graph inputs — data is supplied at evaluate time
        // via feeds, not embedded in the graph.
        inputs.push(t.id);
      }
    } else {
      // Leaf tensor with NO producer op (`t._ctx` is undefined).
      //
      // The only path that produces such a tensor in normal use is `run()` —
      // see `packages/core/src/run.ts`, which builds a bare `new Tensor({...})`
      // and assigns `.data` from `engine.get(id)` so callers can inspect the
      // evaluated values. That result tensor carries a shape, dtype, and a
      // typed-array buffer, but nothing describing how it was produced.
      //
      // Why this branch matters: if the user feeds such a tensor back into a
      // fresh graph op (e.g. `equal(await run(x), ref)` or any follow-up op),
      // trace walks into it looking for its producer so the engine has
      // something to execute. Without a synthetic Constant, the engine would
      // later try to read `registry.get(t.id)`, miss, and throw
      // "Missing expected tensor input" — the user would see a cryptic
      // backend-layer error for what is really just a graph-layer omission.
      //
      // Fix: materialize the `.data` buffer as a synthetic Constant node.
      // This turns the run()-result into a first-class graph input, identical
      // in behavior to `tensor([...])`. The engine's Constant handler
      // (`engine.ts::evaluate`) copies `rawData` into a fresh backend-side
      // storage — no aliasing with the user-visible `.data` buffer, so later
      // mutations on either side don't cross-contaminate.
      //
      // Invariants preserved:
      //   - Tensors with a ctx (`tensor()`, ops, placeholders, parameters)
      //     take the `if (t._ctx)` branch above — never reach here. No
      //     double-materialization.
      //   - `values[t.id]` was populated a few lines up with the tensor's
      //     shape and dtype, so the Constant's output Value is consistent
      //     with what the engine will allocate.
      //   - `visitedOps.has(nodeId)` guards against pushing the same
      //     synthetic node twice when the same leaf is referenced multiple
      //     times in the graph.
      //
      // Error path: if the tensor has neither a producer op NOR `.data`,
      // there is literally nothing to evaluate. Throw at compile time with a
      // pointer to the two ways to fix it, rather than deferring to the
      // engine's generic "missing input" error later.
      if (!t.data) {
        throw new Error(
          `compileGraph: leaf tensor '${t.id}' has no producer op and no .data — ` +
            'nothing to evaluate. Build it with a factory (e.g. `tensor([...])`) ' +
            'or call `await run(t)` first.',
        );
      }
      const nodeId = `node_${t.id}`;
      if (!visitedOps.has(nodeId)) {
        visitedOps.add(nodeId);
        nodes.push({
          id: nodeId,
          op: 'Constant',
          inputs: [],
          outputs: [t.id],
          attributes: { data: t.data as ArrayBufferView },
        });
        values[t.id].producer = nodeId;
      }
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
