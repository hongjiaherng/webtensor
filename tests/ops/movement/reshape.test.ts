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
        await y.equals(
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
        await y.equals(
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
      expect(await y.equals(tensor([11, 22, 33, 44, 55, 66]))).toBe(true);
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
        await y.equals(
          tensor([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
          ]),
        ),
      ).toBe(true);
    });

    it('infers null dim: [24] → [null, 8]', async () => {
      const src = Array.from({ length: 24 }, (_, i) => i + 1);
      const y = await run(contiguous(reshape(tensor(src), [null, 8])), { engine });
      expect(y.shape).toEqual([3, 8]);
      expect(await y.equals(reshape(tensor(src), [3, 8]))).toBe(true);
    });

    it('infers null dim: [2,3,4] → [2, null]', async () => {
      const src = Array.from({ length: 24 }, (_, i) => i + 1);
      const base = reshape(tensor(src), [2, 3, 4]);
      const y = await run(contiguous(reshape(base, [2, null])), { engine });
      expect(y.shape).toEqual([2, 12]);
    });

    it('infers -1 dim: [24] → [-1, 8]', async () => {
      const src = Array.from({ length: 24 }, (_, i) => i + 1);
      const y = await run(contiguous(reshape(tensor(src), [-1, 8])), { engine });
      expect(y.shape).toEqual([3, 8]);
    });

    it('throws on multiple inference placeholders', () => {
      expect(() => reshape(tensor([1, 2, 3, 4]), [null, null])).toThrow(
        /only one dimension can be inferred/,
      );
      expect(() => reshape(tensor([1, 2, 3, 4]), [-1, -1])).toThrow(
        /only one dimension can be inferred/,
      );
    });

    it('throws when inferred dim is not an integer', () => {
      expect(() => reshape(tensor([1, 2, 3, 4, 5]), [null, 2])).toThrow(
        /cannot reshape tensor of size 5/,
      );
    });

    it('throws when explicit shape has wrong total size', () => {
      expect(() => reshape(tensor([1, 2, 3, 4]), [3, 2])).toThrow(
        /cannot reshape tensor of size 4/,
      );
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
      expect(await y.equals(tensor([[10], [26]]))).toBe(true);
    });
  });
});
