import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, transpose, contiguous, slice, reshape, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`contiguous — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('on an already-contiguous tensor (identity)', async () => {
      const y = await run(contiguous(tensor([1, 2, 3, 4])), { engine });
      expect(y.equals(tensor([1, 2, 3, 4]))).toBe(true);
    });

    it('on a transposed view reorders data', async () => {
      const y = await run(
        contiguous(
          transpose(
            tensor([
              [1, 2, 3],
              [4, 5, 6],
            ]),
          ),
        ),
        { engine },
      );
      expect(
        y.equals(
          tensor([
            [1, 4],
            [2, 5],
            [3, 6],
          ]),
        ),
      ).toBe(true);
    });

    it('on a sliced view', async () => {
      const y = await run(
        contiguous(
          slice(
            tensor([
              [1, 2, 3],
              [4, 5, 6],
            ]),
            [1, 0],
            [2, 3],
          ),
        ),
        { engine },
      );
      expect(y.equals(tensor([[4, 5, 6]]))).toBe(true);
    });

    it('reshape on a transposed tensor auto-materializes', async () => {
      const y = await run(
        reshape(
          transpose(
            tensor([
              [1, 2, 3],
              [4, 5, 6],
            ]),
          ),
          [6],
        ),
        { engine },
      );
      expect(y.equals(tensor([1, 4, 2, 5, 3, 6]))).toBe(true);
    });

    it('auto-contiguous reshape matches explicit contiguous + reshape', async () => {
      const data = [
        [1, 2, 3],
        [4, 5, 6],
      ];
      const auto = await run(reshape(transpose(tensor(data)), [6]), { engine });
      const explicit = await run(reshape(contiguous(transpose(tensor(data))), [6]), { engine });
      expect(auto.equals(explicit)).toBe(true);
    });
  });
});
