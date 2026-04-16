import { describe, it, expect, beforeAll } from 'vitest';
import { tensor } from '@webtensor/core';
import { Backend } from '@webtensor/runtime';
import { BACKENDS, runGraph, expectClose } from '../helpers';

const TRANSPOSE_BACKENDS = BACKENDS;

TRANSPOSE_BACKENDS.forEach(({ name, create }) => {
  describe(`Transpose — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('[2,3] → transpose → [3,2]', async () => {
      const out = await runGraph(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
          .transpose()
          .contiguous(),
      );
      expectClose(out, [1, 4, 2, 5, 3, 6]);
    });

    it('[3,3] square matrix', async () => {
      const out = await runGraph(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ])
          .transpose()
          .contiguous(),
      );
      expectClose(out, [1, 4, 7, 2, 5, 8, 3, 6, 9]);
    });

    it('[1,4] row vector → [4,1] col vector', async () => {
      const out = await runGraph(backend, tensor([[10, 20, 30, 40]]).transpose().contiguous());
      expectClose(out, [10, 20, 30, 40]);
    });

    it('[4,1] col vector → [1,4] row vector', async () => {
      const out = await runGraph(
        backend,
        tensor([[10], [20], [30], [40]]).transpose().contiguous(),
      );
      expectClose(out, [10, 20, 30, 40]);
    });

    it('double transpose = identity', async () => {
      const out = await runGraph(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
          .transpose()
          .transpose()
          .contiguous(),
      );
      expectClose(out, [1, 2, 3, 4, 5, 6]);
    });

    it('rank-3 swap last two dims [2,2,3] → [2,3,2]', async () => {
      const out = await runGraph(
        backend,
        tensor([
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
          [
            [7, 8, 9],
            [10, 11, 12],
          ],
        ])
          .transpose()
          .contiguous(),
      );
      expectClose(out, [1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12]);
    });

    it('transpose → matmul (strided input to kernel)', async () => {
      // a.T [3,2] @ identity [2,2] → [3,2] unchanged
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = tensor([
        [1, 0],
        [0, 1],
      ]);
      const out = await runGraph(backend, a.transpose().matmul(b));
      expectClose(out, [1, 4, 2, 5, 3, 6]);
    });

    it('1D input throws', () => {
      expect(() => tensor([1, 2, 3]).transpose()).toThrow();
    });
  });
});
