import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, slice, contiguous, add, run } from '@webtensor/core';
import { relu } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`slice — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[4] → slice([1],[3])', async () => {
      const y = await run(contiguous(slice(tensor([10, 20, 30, 40]), [1], [3])), { engine });
      expect(await y.equals(tensor([20, 30]))).toBe(true);
    });

    it('[2,4] → rows 0:2, cols 1:3', async () => {
      const y = await run(
        contiguous(
          slice(
            tensor([
              [1, 2, 3, 4],
              [5, 6, 7, 8],
            ]),
            [0, 1],
            [2, 3],
          ),
        ),
        { engine },
      );
      expect(
        await y.equals(
          tensor([
            [2, 3],
            [6, 7],
          ]),
        ),
      ).toBe(true);
    });

    it('[3,3] → rows 1:3, cols 0:2', async () => {
      const y = await run(
        contiguous(
          slice(
            tensor([
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
            ]),
            [1, 0],
            [3, 2],
          ),
        ),
        { engine },
      );
      expect(
        await y.equals(
          tensor([
            [4, 5],
            [7, 8],
          ]),
        ),
      ).toBe(true);
    });

    it('slice → add (strided input)', async () => {
      const y = await run(add(slice(tensor([0, 1, 2, 3, 4, 5]), [2], [5]), tensor([10, 10, 10])), {
        engine,
      });
      expect(await y.equals(tensor([12, 13, 14]))).toBe(true);
    });

    it('slice → relu', async () => {
      const y = await run(relu(slice(tensor([-3, -2, -1, 0, 1, 2, 3]), [2], [6])), { engine });
      expect(await y.equals(tensor([0, 0, 1, 2]))).toBe(true);
    });

    it('rank mismatch throws', () => {
      expect(() =>
        slice(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          [0],
          [1],
        ),
      ).toThrow();
    });
  });
});
