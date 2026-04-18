import { describe, it, expect, beforeAll } from 'vitest';
import { compile, tensor, add, mul, matmul, sum } from '@webtensor/core';
import { relu } from '@webtensor/core';
import { Engine, Backend } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`compile() — ${name} Backend`, () => {
    let backend: Backend;

    beforeAll(async () => {
      backend = await create();
    });

    it('single output: traces forward and feeds tensors', async () => {
      const engine = new Engine(backend);
      const forward = await compile(
        ({ x, W, b }) => relu(add(matmul(x, W), b)),
        { x: [1, 3], W: [3, 2], b: [2] },
        { engine },
      );

      const y = await forward({
        x: tensor([[1, -1, 0.5]]),
        W: tensor([
          [1, 0],
          [0, 1],
          [1, 0],
        ]),
        b: tensor([0, 0]),
      });
      // [1,-1,0.5] @ W + 0 → [1.5, -1]; relu → [1.5, 0]
      expect(await y.equals(tensor([[1.5, 0]]))).toBe(true);
    });

    it('feeds also accept raw TypedArray and nested arrays', async () => {
      const engine = new Engine(backend);
      const fn = await compile(({ a, b }) => add(a, b), { a: [3], b: [3] }, { engine });
      const y1 = await fn({ a: new Float32Array([1, 2, 3]), b: new Float32Array([4, 5, 6]) });
      expect(await y1.equals(tensor([5, 7, 9]))).toBe(true);
      const y2 = await fn({ a: [10, 20, 30], b: [1, 2, 3] });
      expect(await y2.equals(tensor([11, 22, 33]))).toBe(true);
    });

    it('is callable many times with different inputs', async () => {
      const engine = new Engine(backend);
      const double = await compile(({ a }) => mul(a, a), { a: [4] }, { engine });
      const y1 = await double({ a: tensor([1, 2, 3, 4]) });
      const y2 = await double({ a: tensor([10, 20, 30, 40]) });
      expect(await y1.equals(tensor([1, 4, 9, 16]))).toBe(true);
      expect(await y2.equals(tensor([100, 400, 900, 1600]))).toBe(true);
    });

    it('multi-output array preserves order', async () => {
      const engine = new Engine(backend);
      const fn = await compile(
        ({ a, b }) => [add(a, b), mul(a, b)] as const,
        { a: [3], b: [3] },
        { engine },
      );
      const [s, prod] = await fn({ a: tensor([1, 2, 3]), b: tensor([4, 5, 6]) });
      expect(await s.equals(tensor([5, 7, 9]))).toBe(true);
      expect(await prod.equals(tensor([4, 10, 18]))).toBe(true);
    });

    it('multi-output record returns keyed tensors', async () => {
      const engine = new Engine(backend);
      const fn = await compile(
        ({ a, b }) => ({ total: add(a, b), dot: sum(mul(a, b)) }),
        { a: [4], b: [4] },
        { engine },
      );
      const { total, dot } = await fn({ a: tensor([1, 2, 3, 4]), b: tensor([1, 1, 1, 1]) });
      expect(await total.equals(tensor([2, 3, 4, 5]))).toBe(true);
      expect(await dot.equals(tensor([10]))).toBe(true);
    });

    it('throws on missing feed key', async () => {
      const engine = new Engine(backend);
      const fn = await compile(({ x }) => relu(x), { x: [3] }, { engine });
      await expect(fn({} as unknown as { x: Float32Array })).rejects.toThrow(/missing feed/);
    });
  });
});
