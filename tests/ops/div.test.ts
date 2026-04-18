import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, div, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`div — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('basic division', async () => {
      const y = await run(div(tensor([10, 20, 9]), tensor([2, 4, 3])), { engine });
      expect(y.equals(tensor([5, 5, 3]))).toBe(true);
    });

    it('scalar broadcast', async () => {
      const y = await run(div(tensor([10, 20, 30]), tensor([10])), { engine });
      expect(y.equals(tensor([1, 2, 3]))).toBe(true);
    });

    it('fractional results', async () => {
      const y = await run(div(tensor([1, 1, 1]), tensor([3, 4, 8])), { engine });
      expect(y.allclose(tensor([1 / 3, 0.25, 0.125]))).toBe(true);
    });

    it('large tensor (1024 elements)', async () => {
      const a = tensor(Array.from({ length: 1024 }, () => 6.0));
      const b = tensor(Array.from({ length: 1024 }, () => 2.0));
      const y = await run(div(a, b), { engine });
      expect(y.equals(tensor(Array.from({ length: 1024 }, () => 3.0)))).toBe(true);
    });

    it('divide by one (identity)', async () => {
      const y = await run(div(tensor([7.5, 3.2, 1.1]), tensor([1, 1, 1])), { engine });
      expect(y.allclose(tensor([7.5, 3.2, 1.1]))).toBe(true);
    });
  });
});
