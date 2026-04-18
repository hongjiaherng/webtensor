import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, placeholder, matmul, add, mul, sub, mean, compileGraph, Engine } from '@webtensor/core';
import { relu } from '@webtensor/nn';
import { CPUBackend } from '@webtensor/backend-cpu';

describe('Placeholder — compile once, feed many', () => {
  let engine: Engine;

  beforeAll(() => {
    engine = new Engine(new CPUBackend());
  });

  it('routes placeholder to graph.inputs (not initializers)', () => {
    const x = placeholder([2, 3]);
    const W = placeholder([3, 4]);
    const y = matmul(x, W);
    const graph = compileGraph([y]);
    expect(graph.inputs).toContain(x.id);
    expect(graph.inputs).toContain(W.id);
    expect(graph.initializers).not.toContain(x.id);
  });

  it('missing feed throws', async () => {
    const x = placeholder([3]);
    const y = relu(x);
    const graph = compileGraph([y]);
    let err: Error | null = null;
    try {
      await engine.evaluate(graph);
    } catch (e) {
      err = e as Error;
    }
    expect(err).not.toBeNull();
    expect(err!.message).toMatch(/Missing feed/);
  });

  it('compile once, feed changing data per step', async () => {
    const x = placeholder([4]);
    const y = mul(x, tensor([2, 2, 2, 2]));
    const graph = compileGraph([y]);

    await engine.evaluate(graph, { [x.id]: new Float32Array([1, 2, 3, 4]) });
    let out = (await engine.get(y.id)) as Float32Array;
    expect(Array.from(out)).toEqual([2, 4, 6, 8]);

    await engine.evaluate(graph, { [x.id]: new Float32Array([10, 20, 30, 40]) });
    out = (await engine.get(y.id)) as Float32Array;
    expect(Array.from(out)).toEqual([20, 40, 60, 80]);
  });

  it('XOR MLP trains with low-level placeholder API', async () => {
    // This test demonstrates the LOW-LEVEL usage of placeholder + compileGraph +
    // evaluate(feeds) + hand-rolled optimizer. For day-to-day training, prefer
    // compile() + grad() + SGD — see tests/graph/train_loop.test.ts.

    const xData = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
    const yData = new Float32Array([0, 1, 1, 0]);

    const hidden = 8;
    // Xavier-scaled init
    function fillRandn(arr: Float32Array, seed: number, std: number) {
      let s = seed;
      const next = () => {
        s ^= s << 13;
        s ^= s >>> 17;
        s ^= s << 5;
        return (s >>> 0) / 0x100000000;
      };
      for (let i = 0; i < arr.length; i += 2) {
        const u1 = Math.max(next(), 1e-12);
        const u2 = next();
        const r = Math.sqrt(-2 * Math.log(u1));
        arr[i] = std * r * Math.cos(2 * Math.PI * u2);
        if (i + 1 < arr.length) arr[i + 1] = std * r * Math.sin(2 * Math.PI * u2);
      }
    }
    const W1 = new Float32Array(2 * hidden);
    fillRandn(W1, 42, 1 / Math.sqrt(2));
    const b1 = new Float32Array(hidden);
    const W2 = new Float32Array(hidden);
    fillRandn(W2, 7, 1 / Math.sqrt(hidden));
    const b2 = new Float32Array(1);

    // Build graph with placeholders for weights + inputs
    const xP = placeholder([4, 2]);
    const yP = placeholder([4, 1]);
    const W1P = placeholder([2, hidden]);
    W1P.requiresGrad = true;
    const b1P = placeholder([hidden]);
    b1P.requiresGrad = true;
    const W2P = placeholder([hidden, 1]);
    W2P.requiresGrad = true;
    const b2P = placeholder([1]);
    b2P.requiresGrad = true;

    const h = relu(add(matmul(xP, W1P), b1P));
    const pred = add(matmul(h, W2P), b2P);
    const diff = sub(pred, yP);
    const loss = mean(mul(diff, diff));
    loss.backward();

    const graph = compileGraph([loss, W1P.grad!, b1P.grad!, W2P.grad!, b2P.grad!]);
    const lr = 0.1;

    let firstLoss = 0;
    let lastLoss = 0;
    for (let step = 0; step < 100; step++) {
      await engine.evaluate(graph, {
        [xP.id]: xData,
        [yP.id]: yData,
        [W1P.id]: W1,
        [b1P.id]: b1,
        [W2P.id]: W2,
        [b2P.id]: b2,
      });
      const lossArr = (await engine.get(loss.id)) as Float32Array;
      if (step === 0) firstLoss = lossArr[0];
      lastLoss = lossArr[0];

      const dW1 = (await engine.get(W1P.grad!.id)) as Float32Array;
      const db1 = (await engine.get(b1P.grad!.id)) as Float32Array;
      const dW2 = (await engine.get(W2P.grad!.id)) as Float32Array;
      const db2 = (await engine.get(b2P.grad!.id)) as Float32Array;
      for (let i = 0; i < W1.length; i++) W1[i] -= lr * dW1[i];
      for (let i = 0; i < b1.length; i++) b1[i] -= lr * db1[i];
      for (let i = 0; i < W2.length; i++) W2[i] -= lr * dW2[i];
      for (let i = 0; i < b2.length; i++) b2[i] -= lr * db2[i];
    }

    expect(firstLoss).toBeGreaterThan(0.1);
    expect(lastLoss).toBeLessThan(firstLoss * 0.7);
  }, 15000);
});
