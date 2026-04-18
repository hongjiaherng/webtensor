import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, add, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`add — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('same-shape vectors', async () => {
      const y = await run(add(tensor([1, 2, 3, 4]), tensor([10, 20, 30, 40])), { engine });
      expect(y.equals(tensor([11, 22, 33, 44]))).toBe(true);
    });

    it('scalar broadcast [1] + [4]', async () => {
      const y = await run(add(tensor([5]), tensor([1, 2, 3, 4])), { engine });
      expect(y.equals(tensor([6, 7, 8, 9]))).toBe(true);
    });

    it('row broadcast [2,3] + [3]', async () => {
      const y = await run(
        add(tensor([[1, 2, 3], [4, 5, 6]]), tensor([10, 20, 30])),
        { engine },
      );
      expect(y.equals(tensor([[11, 22, 33], [14, 25, 36]]))).toBe(true);
    });

    it('rank 3 broadcast [2,2,3] + [3]', async () => {
      const y = await run(
        add(
          tensor([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
          ]),
          tensor([1, 2, 3]),
        ),
        { engine },
      );
      expect(
        y.equals(
          tensor([
            [[2, 4, 6], [5, 7, 9]],
            [[8, 10, 12], [11, 13, 15]],
          ]),
        ),
      ).toBe(true);
    });

    it('large tensor (1024 elements)', async () => {
      const a = tensor(Array.from({ length: 1024 }, () => 1.0));
      const b = tensor(Array.from({ length: 1024 }, () => 2.0));
      const y = await run(add(a, b), { engine });
      const expected = tensor(Array.from({ length: 1024 }, () => 3.0));
      expect(y.equals(expected)).toBe(true);
    });

    it('zeros', async () => {
      const y = await run(add(tensor([0, 0, 0]), tensor([0, 0, 0])), { engine });
      expect(y.equals(tensor([0, 0, 0]))).toBe(true);
    });

    it('negatives cancel', async () => {
      const y = await run(add(tensor([-5, -3, 0, 3]), tensor([5, 3, 0, -3])), { engine });
      expect(y.equals(tensor([0, 0, 0, 0]))).toBe(true);
    });
  });
});

describe('add — shape errors', () => {
  it('incompatible shapes throw at graph-build time', () => {
    const a = tensor([[1, 2, 3], [4, 5, 6]]);
    const b = tensor([[1, 2], [3, 4]]);
    expect(() => add(a, b)).toThrow();
  });
});
