import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, all, any, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`all / any — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('all: rank 1, all true', async () => {
      const y = await run(all(tensor([1, 1, 1], { dtype: 'bool' })), { engine });
      expect(y.dtype).toBe('bool');
      expect(Array.from(y.data!)).toEqual([1]);
    });

    it('all: rank 1, one false', async () => {
      const y = await run(all(tensor([1, 0, 1], { dtype: 'bool' })), { engine });
      expect(Array.from(y.data!)).toEqual([0]);
    });

    it('any: rank 1, all false', async () => {
      const y = await run(any(tensor([0, 0, 0], { dtype: 'bool' })), { engine });
      expect(Array.from(y.data!)).toEqual([0]);
    });

    it('any: rank 1, one true', async () => {
      const y = await run(any(tensor([0, 1, 0], { dtype: 'bool' })), { engine });
      expect(Array.from(y.data!)).toEqual([1]);
    });

    it('all: rank 2, axis=0', async () => {
      const y = await run(
        all(
          tensor(
            [
              [1, 1, 0],
              [1, 0, 1],
            ],
            { dtype: 'bool' },
          ),
          0,
        ),
        { engine },
      );
      expect(y.shape).toEqual([3]);
      expect(Array.from(y.data!)).toEqual([1, 0, 0]);
    });

    it('any: rank 2, axis=1', async () => {
      const y = await run(
        any(
          tensor(
            [
              [0, 0, 0],
              [0, 1, 0],
            ],
            { dtype: 'bool' },
          ),
          1,
        ),
        { engine },
      );
      expect(y.shape).toEqual([2]);
      expect(Array.from(y.data!)).toEqual([0, 1]);
    });

    it('all: keepdim', async () => {
      const y = await run(
        all(
          tensor(
            [
              [1, 1],
              [1, 1],
            ],
            { dtype: 'bool' },
          ),
          1,
          true,
        ),
        { engine },
      );
      expect(y.shape).toEqual([2, 1]);
      expect(Array.from(y.data!)).toEqual([1, 1]);
    });
  });
});
