import { describe, it, beforeAll } from 'vitest';
import { tensor } from '@webtensor/core';
import { Backend } from '@webtensor/runtime';
import { BACKENDS, runGraph, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Reshape — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('[6] → reshape([2,3]) → contiguous', async () => {
      const out = await runGraph(backend, tensor([1, 2, 3, 4, 5, 6]).reshape([2, 3]).contiguous());
      expectClose(out, [1, 2, 3, 4, 5, 6]);
    });

    it('[2,3] → reshape([3,2]) → contiguous', async () => {
      const out = await runGraph(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
          .reshape([3, 2])
          .contiguous(),
      );
      expectClose(out, [1, 2, 3, 4, 5, 6]);
    });

    it('[2,3] → reshape([6]) → add [6]', async () => {
      const out = await runGraph(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
          .reshape([6])
          .add(tensor([10, 20, 30, 40, 50, 60])),
      );
      expectClose(out, [11, 22, 33, 44, 55, 66]);
    });

    it('[2,2,2] → reshape([4,2]) → contiguous', async () => {
      const out = await runGraph(
        backend,
        tensor([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ])
          .reshape([4, 2])
          .contiguous(),
      );
      expectClose(out, [1, 2, 3, 4, 5, 6, 7, 8]);
    });

    it('[4,2] → reshape([2,4]) → matmul([4,1]) → [2,1]', async () => {
      const out = await runGraph(
        backend,
        tensor([
          [1, 2],
          [3, 4],
          [5, 6],
          [7, 8],
        ])
          .reshape([2, 4])
          .matmul(tensor([[1], [1], [1], [1]])),
      );
      expectClose(out, [10, 26]);
    });
  });
});
