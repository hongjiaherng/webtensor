import { describe, it, expect } from 'vitest';
import { tensor, compileGraph, add, mul, matmul, relu } from '../../packages/core/src';
import { Engine } from '../../packages/runtime/src';
import { CPUBackend } from '../../packages/backend-cpu/src';
import { expectClose } from '../helpers';

// All integration tests run on CPU (simplest, synchronous, no setup)

async function run(y: ReturnType<typeof tensor>): Promise<Float32Array> {
  const graph = compileGraph([y]);
  const engine = new Engine(new CPUBackend());
  engine.evaluate(graph);
  return (await engine.get(y.id)) as Float32Array;
}

describe('Graph: relu(a + b)', () => {
  it('zeros negatives after add', async () => {
    const a = tensor([-3, -1, 0, 1, 3]);
    const b = tensor([4, 2, 1, -2, -4]);
    const out = await run(relu(add(a, b)));
    expect(Array.from(out)).toEqual([1, 1, 1, 0, 0]);
  });
});

describe('Graph: (a + b) * c', () => {
  it('correct output and node count', async () => {
    const a = tensor([5.0, 5.5]);
    const b = tensor([3.0, 1.5]);
    const c = tensor([10.0, 10.0]);
    const y = mul(add(a, b), c);

    const graph = compileGraph([y]);
    // 3 Constant nodes + 1 Add + 1 Mul = 5 nodes
    expect(graph.nodes.length).toBe(5);

    const engine = new Engine(new CPUBackend());
    engine.evaluate(graph);
    const out = (await engine.get(y.id)) as Float32Array;
    expect(Array.from(out)).toEqual([80, 70]);
  });
});

describe('Graph: relu(matmul(x,W) + b)', () => {
  it('correct output and node count', async () => {
    const x = tensor([
      [1, -1, 1, -1],
      [-1, 1, -1, 1],
    ]);
    const W = tensor([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1],
    ]);
    const b = tensor([0.5, 0.5, 0.5]);
    const y = relu(add(matmul(x, W), b));

    const graph = compileGraph([y]);
    // 3 Constants (x, W, b) + MatMul + Add + Relu = 6 nodes
    expect(graph.nodes.length).toBe(6);

    const engine = new Engine(new CPUBackend());
    engine.evaluate(graph);
    const out = (await engine.get(y.id)) as Float32Array;
    expectClose(out, [0.5, 0, 0.5, 0.5, 2.5, 0.5]);
  });
});

describe('Graph: two outputs from shared inputs', () => {
  it('both outputs are correct', async () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const y1 = add(a, b);
    const y2 = mul(a, b);

    const graph = compileGraph([y1, y2]);
    expect(graph.outputs.length).toBe(2);

    const engine = new Engine(new CPUBackend());
    engine.evaluate(graph);

    const out1 = (await engine.get(y1.id)) as Float32Array;
    const out2 = (await engine.get(y2.id)) as Float32Array;
    expect(Array.from(out1)).toEqual([5, 7, 9]);
    expect(Array.from(out2)).toEqual([4, 10, 18]);
  });
});

describe('Graph: diamond DAG (shared intermediate)', () => {
  it('mid used twice, node deduplicated', async () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const mid = add(a, b);
    const y = mul(mid, mid);

    const graph = compileGraph([y]);
    // 2 Constants + Add + Mul = 4 nodes (Add appears once)
    expect(graph.nodes.length).toBe(4);

    const engine = new Engine(new CPUBackend());
    engine.evaluate(graph);
    const out = (await engine.get(y.id)) as Float32Array;
    expect(Array.from(out)).toEqual([25, 49, 81]);
  });
});

describe('Graph: metadata', () => {
  it('initializers, outputs, node count for simple add', () => {
    const a = tensor([1]);
    const b = tensor([2]);
    const y = add(a, b);

    const graph = compileGraph([y]);
    // 2 Constant nodes + 1 Add = 3 nodes
    expect(graph.nodes.length).toBe(3);
    // Both leaf tensors have no requiresGrad → become initializers
    expect(graph.initializers.length).toBe(2);
    expect(graph.outputs[0]).toBe(y.id);
  });
});
