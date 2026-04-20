import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, concat, pad, all, any, eq, lt, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

// Cast round-trip coverage lives in tests/ops/elementwise/cast.test.ts. This
// file exercises the other bool-aware kernel paths: concat, pad, strided
// bool inputs via transpose/contiguous, and the compare -> all/any pipeline
// where the bool feeding the reduction comes from a kernel output rather
// than a user-authored constant.

BACKENDS.forEach(({ name, create }) => {
  describe(`bool kernels — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('concat along axis 0', async () => {
      const a = tensor([1, 0, 1], { dtype: 'bool' });
      const b = tensor([0, 1], { dtype: 'bool' });
      const y = await run(concat([a, b], 0), { engine });
      expect(y.dtype).toBe('bool');
      expect(Array.from(y.data!)).toEqual([1, 0, 1, 0, 1]);
    });

    it('concat rank-2 along axis 1', async () => {
      const a = tensor(
        [
          [1, 0],
          [0, 1],
        ],
        { dtype: 'bool' },
      );
      const b = tensor(
        [
          [1, 1, 0],
          [0, 0, 1],
        ],
        { dtype: 'bool' },
      );
      const y = await run(concat([a, b], 1), { engine });
      expect(y.shape).toEqual([2, 5]);
      expect(Array.from(y.data!)).toEqual([1, 0, 1, 1, 0, 0, 1, 0, 0, 1]);
    });

    it('pad with constant fill value 0 (false)', async () => {
      const a = tensor([1, 1, 1], { dtype: 'bool' });
      const y = await run(pad(a, [1, 2], 0), { engine });
      expect(y.dtype).toBe('bool');
      expect(Array.from(y.data!)).toEqual([0, 1, 1, 1, 0, 0]);
    });

    it('pad rank-2 with fill value 1 (true)', async () => {
      const a = tensor(
        [
          [0, 0],
          [0, 0],
        ],
        { dtype: 'bool' },
      );
      const y = await run(pad(a, [1, 0, 0, 1], 1), { engine });
      expect(y.shape).toEqual([3, 3]);
      expect(Array.from(y.data!)).toEqual([1, 1, 1, 0, 0, 1, 0, 0, 1]);
    });

    it('all / any consume bool produced by a comparison (strided chain)', async () => {
      const a = tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = tensor([
        [1.0, 0.0, 3.0],
        [0.0, 5.0, 0.0],
      ]);
      // bool from compare, then reduce
      const allEq = await run(all(eq(a, b)), { engine });
      const anyEq = await run(any(eq(a, b)), { engine });
      expect(allEq.dtype).toBe('bool');
      expect(Array.from(allEq.data!)).toEqual([0]);
      expect(Array.from(anyEq.data!)).toEqual([1]);
    });

    it('reduces bool along axis after compare (non-trivial axis)', async () => {
      const a = tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = tensor([[3.0, 3.0, 3.0]]);
      const mask = lt(a, b); // [[1,1,0],[0,0,0]]
      const rowAny = await run(any(mask, 1), { engine });
      const colAny = await run(any(mask, 0), { engine });
      expect(Array.from(rowAny.data!)).toEqual([1, 0]);
      expect(Array.from(colAny.data!)).toEqual([1, 1, 0]);
    });

    it('transposed bool input round-trips through concat (strided bool)', async () => {
      const a = tensor(
        [
          [1, 0],
          [0, 1],
          [1, 1],
        ],
        { dtype: 'bool' },
      ); // [3, 2]
      const b = tensor(
        [
          [0, 1, 0],
          [1, 0, 1],
        ],
        { dtype: 'bool' },
      ); // [2, 3]
      // a.transpose() is [2, 3] with non-contiguous strides — exercises the
      // strided bool read path in each backend's concat kernel.
      const y = await run(concat([a.transpose(), b], 0), { engine });
      expect(y.shape).toEqual([4, 3]);
      expect(Array.from(y.data!)).toEqual([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]);
    });
  });
});
