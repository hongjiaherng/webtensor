import { describe, it, expect, beforeAll } from 'vitest';
import { pow, tensor, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`pow — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('square (exponent=2)', async () => {
      const y = await run(pow(tensor([1, 2, 3, 4]), 2), { engine });
      expect(y.allclose(tensor([1, 4, 9, 16]))).toBe(true);
    });

    it('cube (exponent=3)', async () => {
      const y = await run(pow(tensor([1, 2, 3]), 3), { engine });
      expect(y.allclose(tensor([1, 8, 27]))).toBe(true);
    });

    it('exponent=0.5 (sqrt)', async () => {
      const y = await run(pow(tensor([4, 9, 16]), 0.5), { engine });
      expect(y.allclose(tensor([2, 3, 4]), { atol: 1e-4 })).toBe(true);
    });
  });
});
