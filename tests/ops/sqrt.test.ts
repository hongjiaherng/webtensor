import { describe, it, beforeAll } from 'vitest';
import { sqrt, tensor, compileGraph } from '@webtensor/core';
import { Engine, Backend } from '@webtensor/runtime';
import { BACKENDS, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Sqrt — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('sqrt of perfect squares', async () => {
      const y = sqrt(tensor([0, 1, 4, 9, 16]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [0, 1, 2, 3, 4]);
    });

    it('sqrt of non-perfect squares', async () => {
      const y = sqrt(tensor([2, 3, 5]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [Math.sqrt(2), Math.sqrt(3), Math.sqrt(5)], 1e-4);
    });
  });
});
