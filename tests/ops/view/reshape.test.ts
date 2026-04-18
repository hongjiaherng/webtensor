import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, reshape, contiguous, add, matmul, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`reshape — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[6] → [2,3]', async () => {
      const y = await run(contiguous(reshape(tensor([1, 2, 3, 4, 5, 6]), [2, 3])), { engine });
      expect(
        y.equals(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
        ),
      ).toBe(true);
    });

    it('[2,3] → [3,2]', async () => {
      const y = await run(
        contiguous(
          reshape(
            tensor([
              [1, 2, 3],
              [4, 5, 6],
            ]),
            [3, 2],
          ),
        ),
        { engine },
      );
      expect(
        y.equals(
          tensor([
            [1, 2],
            [3, 4],
            [5, 6],
          ]),
        ),
      ).toBe(true);
    });

    it('[2,3] → reshape([6]) + add [6]', async () => {
      const y = await run(
        add(
          reshape(
            tensor([
              [1, 2, 3],
              [4, 5, 6],
            ]),
            [6],
          ),
          tensor([10, 20, 30, 40, 50, 60]),
        ),
        { engine },
      );
      expect(y.equals(tensor([11, 22, 33, 44, 55, 66]))).toBe(true);
    });

    it('[2,2,2] → [4,2]', async () => {
      const y = await run(
        contiguous(
          reshape(
            tensor([
              [
                [1, 2],
                [3, 4],
              ],
              [
                [5, 6],
                [7, 8],
              ],
            ]),
            [4, 2],
          ),
        ),
        { engine },
      );
      expect(
        y.equals(
          tensor([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
          ]),
        ),
      ).toBe(true);
    });

    it('reshape → matmul pipeline', async () => {
      const y = await run(
        matmul(
          reshape(
            tensor([
              [1, 2],
              [3, 4],
              [5, 6],
              [7, 8],
            ]),
            [2, 4],
          ),
          tensor([[1], [1], [1], [1]]),
        ),
        { engine },
      );
      expect(y.equals(tensor([[10], [26]]))).toBe(true);
    });
  });
});
