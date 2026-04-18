import { describe, it, expect, beforeAll } from 'vitest';
import { neg, tensor, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`neg — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('negates positive values', async () => {
      const y = await run(neg(tensor([1, 2, 3])), { engine });
      expect(y.equals(tensor([-1, -2, -3]))).toBe(true);
    });

    it('negates negative values', async () => {
      const y = await run(neg(tensor([-1, -2, 0])), { engine });
      expect(y.equals(tensor([1, 2, 0]))).toBe(true);
    });
  });
});
