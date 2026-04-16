import { describe, it, beforeAll } from 'vitest';
import { tanh, tensor, compileGraph } from '@webtensor/core';
import { Engine, Backend } from '@webtensor/runtime';
import { BACKENDS, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Tanh — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('tanh of 0 is 0', async () => {
      const y = tanh(tensor([0]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [0]);
    });

    it('tanh of mixed values', async () => {
      const vals = [-2, -1, 0, 1, 2];
      const y = tanh(tensor(vals));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, vals.map(Math.tanh), 1e-4);
    });
  });
});
