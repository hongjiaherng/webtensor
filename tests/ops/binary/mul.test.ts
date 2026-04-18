import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, mul, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`mul — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('basic rank 1', async () => {
      const y = await run(mul(tensor([2, 3, 4]), tensor([5, 6, 7])), { engine });
      expect(y.equals(tensor([10, 18, 28]))).toBe(true);
    });

    it('scalar broadcast', async () => {
      const y = await run(mul(tensor([1, 2, 3, 4]), tensor([3])), { engine });
      expect(y.equals(tensor([3, 6, 9, 12]))).toBe(true);
    });

    it('row broadcast [2,3] * [3]', async () => {
      const y = await run(
        mul(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          tensor([2, 3, 4]),
        ),
        { engine },
      );
      expect(
        y.equals(
          tensor([
            [2, 6, 12],
            [8, 15, 24],
          ]),
        ),
      ).toBe(true);
    });

    it('rank 3 [2,2,2] * [2]', async () => {
      const y = await run(
        mul(
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
          tensor([10, 100]),
        ),
        { engine },
      );
      expect(
        y.equals(
          tensor([
            [
              [10, 200],
              [30, 400],
            ],
            [
              [50, 600],
              [70, 800],
            ],
          ]),
        ),
      ).toBe(true);
    });

    it('large tensor (1024 elements)', async () => {
      const a = tensor(Array.from({ length: 1024 }, () => 2.0));
      const b = tensor(Array.from({ length: 1024 }, () => 3.0));
      const y = await run(mul(a, b), { engine });
      expect(y.equals(tensor(Array.from({ length: 1024 }, () => 6.0)))).toBe(true);
    });

    it('multiply by zero', async () => {
      const y = await run(mul(tensor([1, 2, 3]), tensor([0, 0, 0])), { engine });
      expect(y.equals(tensor([0, 0, 0]))).toBe(true);
    });

    it('identity (× 1)', async () => {
      const y = await run(mul(tensor([5.5, 2.2, 1.1]), tensor([1, 1, 1])), { engine });
      expect(y.allclose(tensor([5.5, 2.2, 1.1]))).toBe(true);
    });

    it('negative values', async () => {
      const y = await run(mul(tensor([-1, -2, 3]), tensor([4, -5, -6])), { engine });
      expect(y.equals(tensor([-4, 10, -18]))).toBe(true);
    });
  });
});
