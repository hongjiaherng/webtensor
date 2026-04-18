import { describe, it, expect, beforeAll } from 'vitest';
import { sqrt, tensor, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`sqrt — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('sqrt of perfect squares', async () => {
      const y = await run(sqrt(tensor([0, 1, 4, 9, 16])), { engine });
      expect(y.allclose(tensor([0, 1, 2, 3, 4]))).toBe(true);
    });

    it('sqrt of non-perfect squares', async () => {
      const y = await run(sqrt(tensor([2, 3, 5])), { engine });
      expect(
        y.allclose(tensor([Math.sqrt(2), Math.sqrt(3), Math.sqrt(5)]), { atol: 1e-4 }),
      ).toBe(true);
    });
  });
});
