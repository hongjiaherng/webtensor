import { describe, it, beforeAll } from 'vitest';
import { pow, tensor, compileGraph } from '@webtensor/core';
import { Engine, Backend } from '@webtensor/runtime';
import { BACKENDS, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Pow — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('square (exponent=2)', async () => {
      const y = pow(tensor([1, 2, 3, 4]), 2);
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [1, 4, 9, 16]);
    });

    it('cube (exponent=3)', async () => {
      const y = pow(tensor([1, 2, 3]), 3);
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [1, 8, 27]);
    });

    it('exponent=0.5 (sqrt)', async () => {
      const y = pow(tensor([4, 9, 16]), 0.5);
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [2, 3, 4], 1e-4);
    });
  });
});
