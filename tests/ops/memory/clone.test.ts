import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, clone, transpose, add, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`clone — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('preserves values on a contiguous tensor', async () => {
      const y = await run(clone(tensor([1, 2, 3, 4])), { engine });
      expect(y.shape).toEqual([4]);
      expect(await y.equals(tensor([1, 2, 3, 4]))).toBe(true);
    });

    it('materializes a transposed (non-contiguous) view', async () => {
      const y = await run(
        clone(
          transpose(
            tensor([
              [1, 2, 3],
              [4, 5, 6],
            ]),
          ),
        ),
        { engine },
      );
      expect(y.shape).toEqual([3, 2]);
      expect(
        await y.equals(
          tensor([
            [1, 4],
            [2, 5],
            [3, 6],
          ]),
        ),
      ).toBe(true);
    });

    it('clone of a compute result matches the original', async () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([10, 20, 30]);
      const sum = add(a, b);
      const cloned = await run(clone(sum), { engine });
      const original = await run(sum, { engine });
      expect(await cloned.equals(original)).toBe(true);
    });

    it('preserves dtype', async () => {
      const y = await run(clone(tensor([1, 2, 3], { dtype: 'int32' })), { engine });
      expect(y.dtype).toBe('int32');
    });
  });
});
