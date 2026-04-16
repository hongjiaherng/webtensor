import { describe, it, beforeAll } from 'vitest';
import { sigmoid, tensor, compileGraph } from '@webtensor/core';
import { Engine, Backend } from '@webtensor/runtime';
import { BACKENDS, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Sigmoid — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('sigmoid of 0 is 0.5', async () => {
      const y = sigmoid(tensor([0]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [0.5]);
    });

    it('sigmoid of large positive is ~1', async () => {
      const y = sigmoid(tensor([10]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [1 / (1 + Math.exp(-10))], 1e-4);
    });

    it('sigmoid of large negative is ~0', async () => {
      const y = sigmoid(tensor([-10]));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(out, [1 / (1 + Math.exp(10))], 1e-4);
    });

    it('sigmoid of mixed values', async () => {
      const vals = [-2, -1, 0, 1, 2];
      const y = sigmoid(tensor(vals));
      const graph = compileGraph([y]);
      const engine = new Engine(backend);
      await engine.evaluate(graph);
      const out = (await engine.get(y.id)) as Float32Array;
      expectClose(
        out,
        vals.map((v) => 1 / (1 + Math.exp(-v))),
        1e-4,
      );
    });
  });
});
