import { describe, it, expect, beforeAll } from 'vitest';
import { compile, grad, add, matmul, randn, zeros } from '@webtensor/core';
import { relu, mseLoss } from '@webtensor/nn';
import { SGD } from '@webtensor/optim';
import { Engine, Backend } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

// Users work entirely with Tensors — no Float32Array in user code. compile()
// returns evaluated Tensors; SGD takes them directly.

BACKENDS.forEach(({ name, create }) => {
  describe(`Training loop — ${name} Backend`, () => {
    let backend: Backend;

    beforeAll(async () => {
      backend = await create();
    });

    it('XOR MLP: loss decreases (20 steps)', async () => {
      const engine = new Engine(backend);
      const xData = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
      const yData = new Float32Array([0, 1, 1, 0]);

      // Xavier-scaled init to keep activations bounded with lr=0.1.
      const hidden = 8;
      const W1 = randn([2, hidden], { requiresGrad: true, seed: 42, std: 1 / Math.sqrt(2) });
      const b1 = zeros([hidden], { requiresGrad: true });
      const W2 = randn([hidden, 1], { requiresGrad: true, seed: 7, std: 1 / Math.sqrt(hidden) });
      const b2 = zeros([1], { requiresGrad: true });

      const step = await compile(
        ({ x, y }) => {
          const h = relu(add(matmul(x, W1), b1));
          const pred = add(matmul(h, W2), b2);
          const loss = mseLoss(pred, y);
          return {
            loss,
            dW1: grad(loss, W1),
            db1: grad(loss, b1),
            dW2: grad(loss, W2),
            db2: grad(loss, b2),
          };
        },
        { x: [4, 2], y: [4, 1] },
        { engine },
      );

      const opt = new SGD(0.1);

      let firstLoss = 0;
      let lastLoss = 0;
      for (let s = 0; s < 20; s++) {
        const { loss, dW1, db1, dW2, db2 } = await step({ x: xData, y: yData });
        opt.step([W1, b1, W2, b2], [dW1, db1, dW2, db2]);
        const lossVal = (loss.data as Float32Array)[0];
        if (s === 0) firstLoss = lossVal;
        lastLoss = lossVal;
      }
      // Prove the training loop works — loss drops. Full convergence is left
      // to integration benchmarks.
      expect(firstLoss).toBeGreaterThan(0.1);
      expect(lastLoss).toBeLessThan(firstLoss);
    }, 15000);

    it('forward-only with trainable params captured works', async () => {
      const engine = new Engine(backend);
      const W = randn([2, 3], { requiresGrad: true, seed: 1 });
      const forward = await compile(({ x }) => matmul(x, W), { x: [4, 2] }, { engine });
      const xData = new Float32Array([1, 0, 0, 1, 1, 1, -1, -1]);
      const y = await forward({ x: xData });
      expect(y.shape).toEqual([4, 3]);
      expect(y.data!.length).toBe(12);
    });

    it('opt.step mutates param .data in place', async () => {
      const engine = new Engine(backend);
      const W = randn([3], { requiresGrad: true, seed: 99, std: 0.5 });
      const before = Array.from(W.data as Float32Array);

      const step = await compile(
        ({ x, y }) => {
          const loss = mseLoss(add(x, W), y);
          return { loss, dW: grad(loss, W) };
        },
        { x: [3], y: [3] },
        { engine },
      );

      const opt = new SGD(0.05);
      for (let i = 0; i < 3; i++) {
        const { dW } = await step({
          x: new Float32Array([0, 0, 0]),
          y: new Float32Array([1, 2, 3]),
        });
        opt.step([W], [dW]);
      }
      const after = Array.from(W.data as Float32Array);
      expect(after).not.toEqual(before);
      // W was being pushed toward [1, 2, 3]:
      for (let i = 0; i < 3; i++) {
        expect(Math.abs(after[i] - (i + 1))).toBeLessThan(Math.abs(before[i] - (i + 1)));
      }
    }, 30000);
  });
});

describe('compile() — grad error handling', () => {
  it('grad(loss, paramNotUsed) throws clearly', async () => {
    const { CPUBackend } = await import('@webtensor/backend-cpu');
    const engine = new Engine(await CPUBackend.create());
    const W = randn([2, 3], { requiresGrad: true, seed: 5 });
    const Wunused = randn([2, 3], { requiresGrad: true, seed: 6 });
    await expect(
      compile(
        ({ x, y }) => {
          const loss = mseLoss(matmul(x, W), y);
          return { loss, dW: grad(loss, W), dUnused: grad(loss, Wunused) };
        },
        { x: [4, 2], y: [4, 3] },
        { engine },
      ),
    ).rejects.toThrow(/param has no gradient/);
  });
});
