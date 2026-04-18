import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, transpose, contiguous, matmul, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`transpose — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[2,3] → [3,2]', async () => {
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

    it('[3,3] square matrix', async () => {
      const y = await run(
        contiguous(
          transpose(
            tensor([
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
            ]),
          ),
        ),
        { engine },
      );
      expect(
        y.equals(
          tensor([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
          ]),
        ),
      ).toBe(true);
    });

    it('[1,4] row → [4,1] col', async () => {
      const y = await run(contiguous(transpose(tensor([[10, 20, 30, 40]]))), { engine });
      expect(y.equals(tensor([[10], [20], [30], [40]]))).toBe(true);
    });

    it('[4,1] col → [1,4] row', async () => {
      const y = await run(contiguous(transpose(tensor([[10], [20], [30], [40]]))), { engine });
      expect(y.equals(tensor([[10, 20, 30, 40]]))).toBe(true);
    });

    it('double transpose = identity', async () => {
      const y = await run(
        contiguous(
          transpose(
            transpose(
              tensor([
                [1, 2, 3],
                [4, 5, 6],
              ]),
            ),
          ),
        ),
        { engine },
      );
      expect(
        y.equals(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
        ),
      ).toBe(true);
    });

    it('rank-3 swap last two dims [2,2,3] → [2,3,2]', async () => {
      const y = await run(
        contiguous(
          transpose(
            tensor([
              [
                [1, 2, 3],
                [4, 5, 6],
              ],
              [
                [7, 8, 9],
                [10, 11, 12],
              ],
            ]),
          ),
        ),
        { engine },
      );
      expect(
        y.equals(
          tensor([
            [
              [1, 4],
              [2, 5],
              [3, 6],
            ],
            [
              [7, 10],
              [8, 11],
              [9, 12],
            ],
          ]),
        ),
      ).toBe(true);
    });

    it('transpose → matmul with strided input', async () => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = tensor([
        [1, 0],
        [0, 1],
      ]);
      const y = await run(matmul(transpose(a), b), { engine });
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

    it('1D input throws', () => {
      expect(() => transpose(tensor([1, 2, 3]))).toThrow();
    });
  });
});
