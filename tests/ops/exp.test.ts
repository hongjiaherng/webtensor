import { describe, it, beforeAll } from 'vitest';
import { exp, tensor, compileGraph } from '@webtensor/core';
import { Engine, Backend } from '@webtensor/runtime';
import { BACKENDS, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Exp — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('exp of zero is 1', async () => {
      const y = exp(tensor([0, 0, 0]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [1, 1, 1]);
    });

    it('exp of 1 is e', async () => {
      const y = exp(tensor([1]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [Math.E], 1e-4);
    });

    it('exp of negative values', async () => {
      const y = exp(tensor([-1, -2]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [Math.exp(-1), Math.exp(-2)], 1e-4);
    });
  });
});
