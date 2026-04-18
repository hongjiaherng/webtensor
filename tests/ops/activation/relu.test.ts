import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, run } from '@webtensor/core';
import { relu } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`relu — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('mixed values', async () => {
      const y = await run(relu(tensor([-2, -1, 0, 1, 2])), { engine });
      expect(await y.equals(tensor([0, 0, 0, 1, 2]))).toBe(true);
    });

    it('all positive (pass-through)', async () => {
      const y = await run(relu(tensor([0.1, 1.5, 3.0, 100.0])), { engine });
      expect(await y.allclose(tensor([0.1, 1.5, 3.0, 100.0]))).toBe(true);
    });

    it('all negative (zero out)', async () => {
      const y = await run(relu(tensor([-0.1, -1.5, -100.0])), { engine });
      expect(await y.equals(tensor([0, 0, 0]))).toBe(true);
    });

    it('all zeros', async () => {
      const y = await run(relu(tensor([0, 0, 0, 0])), { engine });
      expect(await y.equals(tensor([0, 0, 0, 0]))).toBe(true);
    });

    it('large tensor (1024 elements: 512 neg + 512 pos)', async () => {
      const data = [
        ...Array.from({ length: 512 }, () => -1.0),
        ...Array.from({ length: 512 }, () => 1.0),
      ];
      const expected = tensor([
        ...Array.from({ length: 512 }, () => 0),
        ...Array.from({ length: 512 }, () => 1.0),
      ]);
      const y = await run(relu(tensor(data)), { engine });
      expect(await y.equals(expected)).toBe(true);
    });

    it('boundary at zero', async () => {
      const y = await run(relu(tensor([-0.0001, 0.0, 0.0001])), { engine });
      expect(await y.allclose(tensor([0, 0, 0.0001]))).toBe(true);
    });
  });
});
