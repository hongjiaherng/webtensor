import { describe, it, beforeAll } from 'vitest';
import { tensor } from '@webtensor/core';
import { Backend } from '@webtensor/runtime';
import { BACKENDS, runGraph, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Contiguous — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('contiguous on already-contiguous tensor (identity)', async () => {
      const out = await runGraph(backend, tensor([1, 2, 3, 4]).contiguous());
      expectClose(out, [1, 2, 3, 4]);
    });

    it('contiguous on transposed view reorders data', async () => {
      // [[1,2,3],[4,5,6]] transposed → [[1,4],[2,5],[3,6]]
      // flat contiguous: [1,4,2,5,3,6]
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

    it('contiguous on sliced view', async () => {
      // [[1,2,3],[4,5,6]] slice rows [1,2) cols [0,3) → [[4,5,6]]
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const sliced = a.slice([1, 0], [2, 3]);
      const out = await runGraph(backend, sliced.contiguous());
      expectClose(out, [4, 5, 6]);
    });

    it('reshape after transpose (auto-contiguous copy)', async () => {
      // [[1,2,3],[4,5,6]] transposed → [[1,4],[2,5],[3,6]] (non-contiguous)
      // reshape to [6] should auto-copy → [1,4,2,5,3,6]
      const out = await runGraph(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
          .transpose()
          .reshape([6]),
      );
      expectClose(out, [1, 4, 2, 5, 3, 6]);
    });

    it('reshape after transpose matches explicit contiguous + reshape', async () => {
      // Both paths should produce the same result
      const data = [
        [1, 2, 3],
        [4, 5, 6],
      ];
      const autoPath = await runGraph(
        backend,
        tensor(data).transpose().reshape([6]),
      );
      const explicitPath = await runGraph(
        backend,
        tensor(data).transpose().contiguous().reshape([6]),
      );
      expectClose(autoPath, Array.from(explicitPath));
    });
  });
});
