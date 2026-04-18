import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, concat, add, mul, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`concat — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('concat along axis 0 (default)', async () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([
        [5, 6],
        [7, 8],
      ]);
      const y = await run(concat([a, b]), { engine });
      expect(y.shape).toEqual([4, 2]);
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

    it('concat along axis 1', async () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([
        [5, 6, 7],
        [8, 9, 10],
      ]);
      const y = await run(concat([a, b], 1), { engine });
      expect(y.shape).toEqual([2, 5]);
      expect(
        await y.equals(
          tensor([
            [1, 2, 5, 6, 7],
            [3, 4, 8, 9, 10],
          ]),
        ),
      ).toBe(true);
    });

    it('concat three tensors along axis -1', async () => {
      const a = tensor([[1], [2]]);
      const b = tensor([
        [3, 4],
        [5, 6],
      ]);
      const c = tensor([[7], [8]]);
      const y = await run(concat([a, b, c], -1), { engine });
      expect(y.shape).toEqual([2, 4]);
      expect(
        await y.equals(
          tensor([
            [1, 3, 4, 7],
            [2, 5, 6, 8],
          ]),
        ),
      ).toBe(true);
    });

    it('concat rank-1 tensors', async () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5]);
      const c = tensor([6]);
      const y = await run(concat([a, b, c]), { engine });
      expect(await y.equals(tensor([1, 2, 3, 4, 5, 6]))).toBe(true);
    });

    it('concat rank-3 along middle axis', async () => {
      // [2,1,3] + [2,2,3] → [2,3,3]
      const a = tensor([[[1, 2, 3]], [[4, 5, 6]]]);
      const b = tensor([
        [
          [7, 8, 9],
          [10, 11, 12],
        ],
        [
          [13, 14, 15],
          [16, 17, 18],
        ],
      ]);
      const y = await run(concat([a, b], 1), { engine });
      expect(y.shape).toEqual([2, 3, 3]);
      expect(
        await y.equals(
          tensor([
            [
              [1, 2, 3],
              [7, 8, 9],
              [10, 11, 12],
            ],
            [
              [4, 5, 6],
              [13, 14, 15],
              [16, 17, 18],
            ],
          ]),
        ),
      ).toBe(true);
    });

    it('concat int32 tensors preserves dtype', async () => {
      const a = tensor([1, 2], { dtype: 'int32' });
      const b = tensor([3, 4, 5], { dtype: 'int32' });
      const y = await run(concat([a, b]), { engine });
      expect(y.dtype).toBe('int32');
      expect(await y.equals(tensor([1, 2, 3, 4, 5], { dtype: 'int32' }))).toBe(true);
    });

    it('concat handles non-contiguous inputs (post-transpose)', async () => {
      // a.T is [[1,3],[2,4]], b.T is [[5,7],[6,8]]; concat along axis 0 → 4x2
      const a = tensor([
        [1, 2],
        [3, 4],
      ]).transpose();
      const b = tensor([
        [5, 6],
        [7, 8],
      ]).transpose();
      const y = await run(concat([a, b]), { engine });
      expect(y.shape).toEqual([4, 2]);
      expect(
        await y.equals(
          tensor([
            [1, 3],
            [2, 4],
            [5, 7],
            [6, 8],
          ]),
        ),
      ).toBe(true);
    });

    it('single-input concat is identity', async () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const y = await run(concat([a]), { engine });
      expect(await y.equals(a)).toBe(true);
    });

    it('chains with other ops', async () => {
      const a = tensor([1, 2]);
      const b = tensor([3, 4]);
      const c = tensor([10, 20, 30, 40]);
      const y = await run(add(concat([a, b]), c), { engine });
      expect(await y.equals(tensor([11, 22, 33, 44]))).toBe(true);
    });

    it('backward splits grad into axis-slices', async () => {
      const a = tensor([1.0, 2.0, 3.0], { requiresGrad: true });
      const b = tensor([4.0, 5.0], { requiresGrad: true });
      const y = concat([a, b]);
      // y.sum().backward() → grad at y is all 1s; a gets [1,1,1], b gets [1,1].
      y.sum().backward();
      const aGrad = await run(a.grad!, { engine });
      const bGrad = await run(b.grad!, { engine });
      expect(await aGrad.equals(tensor([1, 1, 1]))).toBe(true);
      expect(await bGrad.equals(tensor([1, 1]))).toBe(true);
    });

    it('backward with weighted loss', async () => {
      const a = tensor([1.0, 2.0, 3.0], { requiresGrad: true });
      const b = tensor([4.0, 5.0], { requiresGrad: true });
      const y = concat([a, b]);
      const loss = mul(y, tensor([10, 20, 30, 40, 50]));
      loss.sum().backward();
      const aGrad = await run(a.grad!, { engine });
      const bGrad = await run(b.grad!, { engine });
      expect(await aGrad.equals(tensor([10, 20, 30]))).toBe(true);
      expect(await bGrad.equals(tensor([40, 50]))).toBe(true);
    });
  });
});

describe('concat — input validation', () => {
  it('rejects empty input list', () => {
    expect(() => concat([])).toThrow(/at least one/);
  });

  it('rejects mismatched ranks', () => {
    expect(() => concat([tensor([1, 2]), tensor([[3, 4]])])).toThrow(/rank/);
  });

  it('rejects mismatched dtypes', () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4], { dtype: 'int32' });
    expect(() => concat([a, b])).toThrow(/dtype/);
  });

  it('rejects mismatched non-axis dim', () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]); // [2,2]
    const b = tensor([[5, 6, 7]]); // [1,3]
    expect(() => concat([a, b], 0)).toThrow(/dim 1/);
  });

  it('rejects out-of-range axis', () => {
    const a = tensor([[1, 2]]);
    expect(() => concat([a, a], 5)).toThrow(/out of range/);
  });

  it('negative axis is normalized', () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = concat([a, a], -1);
    expect(y.shape).toEqual([2, 4]);
  });
});
