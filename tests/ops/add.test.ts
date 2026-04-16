import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, add } from '@webtensor/core';
import { Backend } from '@webtensor/runtime';
import { BACKENDS, runBinary } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Add — ${name} Backend`, () => {
    let backend: Backend;

    beforeAll(async () => {
      backend = await create();
    });

    it('same-shape vector (rank 1)', async () => {
      const out = await runBinary(backend, add, [1, 2, 3, 4], [10, 20, 30, 40]);
      expect(Array.from(out)).toEqual([11, 22, 33, 44]);
    });

    it('scalar broadcast [1] + [4]', async () => {
      const out = await runBinary(backend, add, [5], [1, 2, 3, 4]);
      expect(Array.from(out)).toEqual([6, 7, 8, 9]);
    });

    it('row broadcast [2,3] + [3]', async () => {
      const out = await runBinary(
        backend,
        add,
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [10, 20, 30],
      );
      expect(Array.from(out)).toEqual([11, 22, 33, 14, 25, 36]);
    });

    it('rank 3: [2,2,3] + [3]', async () => {
      const out = await runBinary(
        backend,
        add,
        [
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
          [
            [7, 8, 9],
            [10, 11, 12],
          ],
        ],
        [1, 2, 3],
      );
      expect(Array.from(out)).toEqual([2, 4, 6, 5, 7, 9, 8, 10, 12, 11, 13, 15]);
    });

    it('large tensor (1024 elements)', async () => {
      const a = Array.from({ length: 1024 }, () => 1.0);
      const b = Array.from({ length: 1024 }, () => 2.0);
      const out = await runBinary(backend, add, a, b);
      expect(out.length).toBe(1024);
      expect(out.every((v) => v === 3.0)).toBe(true);
    });

    it('zeros', async () => {
      const out = await runBinary(backend, add, [0, 0, 0], [0, 0, 0]);
      expect(Array.from(out)).toEqual([0, 0, 0]);
    });

    it('negatives cancel', async () => {
      const out = await runBinary(backend, add, [-5, -3, 0, 3], [5, 3, 0, -3]);
      expect(Array.from(out)).toEqual([0, 0, 0, 0]);
    });
  });
});

describe('Add — shape error (no backend)', () => {
  it('incompatible shapes throw during graph construction', () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]); // [2,3]
    const b = tensor([
      [1, 2],
      [3, 4],
    ]); // [2,2]
    expect(() => add(a, b)).toThrow();
  });
});
