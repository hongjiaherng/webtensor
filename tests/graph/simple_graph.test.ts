import { describe, it, expect } from 'vitest';
import { tensor, compileGraph, add, mul, matmul, run } from '@webtensor/core';
import { relu } from '@webtensor/nn';
import { Engine } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';

const engine = new Engine(new CPUBackend());

describe('Graph: relu(a + b)', () => {
  it('zeros negatives after add', async () => {
    const y = await run(
      relu(add(tensor([-3, -1, 0, 1, 3]), tensor([4, 2, 1, -2, -4]))),
      { engine },
    );
    expect(y.equals(tensor([1, 1, 1, 0, 0]))).toBe(true);
  });
});

describe('Graph: (a + b) * c', () => {
  it('correct output and node count', async () => {
    const a = tensor([5.0, 5.5]);
    const b = tensor([3.0, 1.5]);
    const c = tensor([10.0, 10.0]);
    const y = mul(add(a, b), c);

    // 3 Constant nodes + 1 Add + 1 Mul = 5 nodes
    expect(compileGraph([y]).nodes.length).toBe(5);

    const out = await run(y, { engine });
    expect(out.equals(tensor([80, 70]))).toBe(true);
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

    // 3 Constants (x, W, b) + MatMul + Add + Relu = 6 nodes
    expect(compileGraph([y]).nodes.length).toBe(6);

    const out = await run(y, { engine });
    expect(out.allclose(tensor([[0.5, 0, 0.5], [0.5, 2.5, 0.5]]))).toBe(true);
  });
});

describe('Graph: two outputs from shared inputs', () => {
  it('both outputs are correct', async () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const [y1, y2] = await run([add(a, b), mul(a, b)], { engine });
    expect(y1.equals(tensor([5, 7, 9]))).toBe(true);
    expect(y2.equals(tensor([4, 10, 18]))).toBe(true);
  });
});

describe('Graph: diamond DAG (shared intermediate)', () => {
  it('mid used twice, node deduplicated', async () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const mid = add(a, b);
    const y = mul(mid, mid);

    // 2 Constants + Add + Mul = 4 nodes (Add appears once)
    expect(compileGraph([y]).nodes.length).toBe(4);

    const out = await run(y, { engine });
    expect(out.equals(tensor([25, 49, 81]))).toBe(true);
  });
});

describe('Graph: metadata', () => {
  it('initializers, outputs, node count for simple add', () => {
    const a = tensor([1]);
    const b = tensor([2]);
    const y = add(a, b);

    const graph = compileGraph([y]);
    expect(graph.nodes.length).toBe(3); // 2 Constants + 1 Add
    expect(graph.initializers.length).toBe(2);
    expect(graph.outputs[0]).toBe(y.id);
  });
});
