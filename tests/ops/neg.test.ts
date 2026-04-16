import { describe, it, beforeAll } from 'vitest';
import { neg, tensor, compileGraph } from '@webtensor/core';
import { Engine, Backend } from '@webtensor/runtime';
import { BACKENDS, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Neg — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('negates positive values', async () => {
      const a = tensor([1, 2, 3]);
      const y = neg(a);
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [-1, -2, -3]);
    });

    it('negates negative values', async () => {
      const a = tensor([-1, -2, 0]);
      const y = neg(a);
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [1, 2, 0]);
    });
  });
});
