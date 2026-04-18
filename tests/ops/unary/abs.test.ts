import { describe, it, expect, beforeAll } from 'vitest';
import { abs, tensor, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`abs — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('abs of mixed values', async () => {
      const y = await run(abs(tensor([-3, -1, 0, 1, 3])), { engine });
      expect(y.equals(tensor([3, 1, 0, 1, 3]))).toBe(true);
    });
  });
});
