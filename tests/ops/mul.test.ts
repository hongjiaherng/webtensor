import { describe, it, expect, beforeAll } from 'vitest';
import { mul } from '@webtensor/core';
import { Backend } from '@webtensor/runtime';
import { BACKENDS, runBinary, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Mul — ${name} Backend`, () => {
    let backend: Backend;

    beforeAll(async () => {
      backend = await create();
    });

    it('basic (rank 1)', async () => {
      const out = await runBinary(backend, mul, [2, 3, 4], [5, 6, 7]);
      expect(Array.from(out)).toEqual([10, 18, 28]);
    });

    it('scalar broadcast [4] * [1]', async () => {
      const out = await runBinary(backend, mul, [1, 2, 3, 4], [3]);
      expect(Array.from(out)).toEqual([3, 6, 9, 12]);
    });

    it('row broadcast [2,3] * [3]', async () => {
      const out = await runBinary(
        backend,
        mul,
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [2, 3, 4],
      );
      expect(Array.from(out)).toEqual([2, 6, 12, 8, 15, 24]);
    });

    it('rank 3: [2,2,2] * [2]', async () => {
      const out = await runBinary(
        backend,
        mul,
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ],
        [10, 100],
      );
      expect(Array.from(out)).toEqual([10, 200, 30, 400, 50, 600, 70, 800]);
    });

    it('large tensor (1024 elements)', async () => {
      const a = Array.from({ length: 1024 }, () => 2.0);
      const b = Array.from({ length: 1024 }, () => 3.0);
      const out = await runBinary(backend, mul, a, b);
      expect(out.length).toBe(1024);
      expect(out.every((v) => v === 6.0)).toBe(true);
    });

    it('multiply by zero', async () => {
      const out = await runBinary(backend, mul, [1, 2, 3], [0, 0, 0]);
      expect(Array.from(out)).toEqual([0, 0, 0]);
    });

    it('identity (× 1)', async () => {
      const out = await runBinary(backend, mul, [5.5, 2.2, 1.1], [1, 1, 1]);
      expectClose(out, [5.5, 2.2, 1.1]);
    });

    it('negative values', async () => {
      const out = await runBinary(backend, mul, [-1, -2, 3], [4, -5, -6]);
      expect(Array.from(out)).toEqual([-4, 10, -18]);
    });
  });
});
