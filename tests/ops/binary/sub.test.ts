import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, sub, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`sub — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('basic subtraction', async () => {
      const y = await run(sub(tensor([10, 20, 30]), tensor([1, 2, 3])), { engine });
      expect(y.equals(tensor([9, 18, 27]))).toBe(true);
    });

    it('scalar broadcast', async () => {
      const y = await run(sub(tensor([10, 20, 30]), tensor([5])), { engine });
      expect(y.equals(tensor([5, 15, 25]))).toBe(true);
    });

    it('negative result', async () => {
      const y = await run(sub(tensor([1, 2, 3]), tensor([4, 5, 6])), { engine });
      expect(y.equals(tensor([-3, -3, -3]))).toBe(true);
    });

    it('large tensor (1024 elements)', async () => {
      const a = tensor(Array.from({ length: 1024 }, () => 5.0));
      const b = tensor(Array.from({ length: 1024 }, () => 3.0));
      const y = await run(sub(a, b), { engine });
      expect(y.equals(tensor(Array.from({ length: 1024 }, () => 2.0)))).toBe(true);
    });

    it('self-subtraction yields zeros', async () => {
      const y = await run(sub(tensor([1.5, 2.5, 3.5]), tensor([1.5, 2.5, 3.5])), { engine });
      expect(y.allclose(tensor([0, 0, 0]))).toBe(true);
    });
  });
});
