import { describe, it, expect, beforeAll } from 'vitest';
import { log, tensor, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`log — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('log(1) = 0', async () => {
      const y = await run(log(tensor([1, 1, 1])), { engine });
      expect(y.allclose(tensor([0, 0, 0]))).toBe(true);
    });

    it('log(e) = 1', async () => {
      const y = await run(log(tensor([Math.E])), { engine });
      expect(y.allclose(tensor([1]), { atol: 1e-4 })).toBe(true);
    });

    it('log of positive values', async () => {
      const y = await run(log(tensor([2, 10, 100])), { engine });
      expect(
        y.allclose(tensor([Math.log(2), Math.log(10), Math.log(100)]), { atol: 1e-4 }),
      ).toBe(true);
    });
  });
});
