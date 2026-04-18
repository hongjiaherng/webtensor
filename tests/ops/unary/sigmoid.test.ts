import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, run } from '@webtensor/core';
import { sigmoid } from '@webtensor/nn';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`sigmoid — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('sigmoid(0) = 0.5', async () => {
      const y = await run(sigmoid(tensor([0])), { engine });
      expect(y.allclose(tensor([0.5]))).toBe(true);
    });

    it('sigmoid(10) ≈ 1', async () => {
      const y = await run(sigmoid(tensor([10])), { engine });
      expect(y.allclose(tensor([1 / (1 + Math.exp(-10))]), { atol: 1e-4 })).toBe(true);
    });

    it('sigmoid(-10) ≈ 0', async () => {
      const y = await run(sigmoid(tensor([-10])), { engine });
      expect(y.allclose(tensor([1 / (1 + Math.exp(10))]), { atol: 1e-4 })).toBe(true);
    });

    it('sigmoid of mixed values', async () => {
      const vals = [-2, -1, 0, 1, 2];
      const y = await run(sigmoid(tensor(vals)), { engine });
      expect(y.allclose(tensor(vals.map((v) => 1 / (1 + Math.exp(-v)))), { atol: 1e-4 })).toBe(
        true,
      );
    });
  });
});
