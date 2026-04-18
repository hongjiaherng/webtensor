import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, sum, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`sum — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('rank 1: sum all elements', async () => {
      const y = await run(sum(tensor([1, 2, 3, 4])), { engine });
      expect(y.equals(tensor([10]))).toBe(true);
    });

    it('rank 2: sum all', async () => {
      const y = await run(sum(tensor([[1, 2, 3], [4, 5, 6]])), { engine });
      expect(y.equals(tensor([21]))).toBe(true);
    });

    it('rank 2: axis 0', async () => {
      const y = await run(sum(tensor([[1, 2, 3], [4, 5, 6]]), 0), { engine });
      expect(y.equals(tensor([5, 7, 9]))).toBe(true);
    });

    it('rank 2: axis 1', async () => {
      const y = await run(sum(tensor([[1, 2, 3], [4, 5, 6]]), 1), { engine });
      expect(y.equals(tensor([6, 15]))).toBe(true);
    });

    it('rank 2: axis -1 (negative)', async () => {
      const y = await run(sum(tensor([[1, 2, 3], [4, 5, 6]]), -1), { engine });
      expect(y.equals(tensor([6, 15]))).toBe(true);
    });

    it('rank 2: axis 0 keepdim=true → shape [1,3]', async () => {
      const y = await run(sum(tensor([[1, 2, 3], [4, 5, 6]]), 0, true), { engine });
      expect(y.equals(tensor([[5, 7, 9]]))).toBe(true);
    });

    it('rank 3: sum axes [0, 2]', async () => {
      const y = await run(
        sum(tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), [0, 2]),
        { engine },
      );
      expect(y.equals(tensor([14, 22]))).toBe(true);
    });

    it('rank 3: axis 1 → shape [2,2]', async () => {
      const y = await run(sum(tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), 1), { engine });
      expect(y.equals(tensor([[4, 6], [12, 14]]))).toBe(true);
    });

    it('rank 4: sum all', async () => {
      const y = await run(
        sum(tensor([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]])),
        { engine },
      );
      expect(y.equals(tensor([36]))).toBe(true);
    });

    it('rank 4: axes [1,3]', async () => {
      const y = await run(
        sum(tensor([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]), [1, 3]),
        { engine },
      );
      expect(y.equals(tensor([[3, 7], [11, 15]]))).toBe(true);
    });

    it('single element → itself', async () => {
      const y = await run(sum(tensor([[42]])), { engine });
      expect(y.equals(tensor([42]))).toBe(true);
    });

    it('empty axes (reduce none): returns original flat', async () => {
      const y = await run(sum(tensor([[1, 2], [3, 4]]), []), { engine });
      expect(y.equals(tensor([[1, 2], [3, 4]]))).toBe(true);
    });
  });
});

BACKENDS.forEach(({ name, create }) => {
  describe(`sum — autograd — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('gradient of sum is all-ones', async () => {
      const a = tensor([[1, 2], [3, 4]], { requiresGrad: true });
      const y = sum(a);
      y.backward();
      const g = await run(a.grad!, { engine });
      expect(g.allclose(tensor([[1, 1], [1, 1]]))).toBe(true);
    });

    it('gradient of sum axis 0 broadcasts back to input shape', async () => {
      const a = tensor([[1, 2, 3], [4, 5, 6]], { requiresGrad: true });
      const y = sum(a, 0);
      y.backward();
      const g = await run(a.grad!, { engine });
      expect(g.allclose(tensor([[1, 1, 1], [1, 1, 1]]))).toBe(true);
    });
  });
});
