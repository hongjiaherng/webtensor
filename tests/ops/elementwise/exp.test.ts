import { describe, it, expect, beforeAll } from 'vitest';
import { exp, tensor, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`exp — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('exp(0) = 1', async () => {
      const y = await run(exp(tensor([0, 0, 0])), { engine });
      expect(await y.allclose(tensor([1, 1, 1]))).toBe(true);
    });

    it('exp(1) = e', async () => {
      const y = await run(exp(tensor([1])), { engine });
      expect(await y.allclose(tensor([Math.E]), { atol: 1e-4 })).toBe(true);
    });

    it('exp of negative values', async () => {
      const y = await run(exp(tensor([-1, -2])), { engine });
      expect(await y.allclose(tensor([Math.exp(-1), Math.exp(-2)]), { atol: 1e-4 })).toBe(true);
    });
  });
});
