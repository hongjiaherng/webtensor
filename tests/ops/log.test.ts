import { describe, it, beforeAll } from 'vitest';
import { log, tensor, compileGraph } from '@webtensor/core';
import { Engine, Backend } from '@webtensor/runtime';
import { BACKENDS, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Log — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('log of 1 is 0', async () => {
      const y = log(tensor([1, 1, 1]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [0, 0, 0]);
    });

    it('log of e is 1', async () => {
      const y = log(tensor([Math.E]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [1], 1e-4);
    });

    it('log of positive values', async () => {
      const y = log(tensor([2, 10, 100]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [Math.log(2), Math.log(10), Math.log(100)], 1e-4);
    });
  });
});
