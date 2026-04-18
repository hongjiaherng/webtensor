import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, run } from '@webtensor/core';
import { tanh } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`tanh — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('tanh(0) = 0', async () => {
      const y = await run(tanh(tensor([0])), { engine });
      expect(await y.allclose(tensor([0]))).toBe(true);
    });

    it('tanh of mixed values', async () => {
      const vals = [-2, -1, 0, 1, 2];
      const y = await run(tanh(tensor(vals)), { engine });
      expect(await y.allclose(tensor(vals.map(Math.tanh)), { atol: 1e-4 })).toBe(true);
    });
  });
});
