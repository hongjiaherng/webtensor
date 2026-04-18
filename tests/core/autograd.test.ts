import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, matmul, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

// Forward + backward of a simple matmul, verified across all three backends.

BACKENDS.forEach(({ name, create }) => {
  describe(`autograd — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('matmul(a, b) gradients follow the chain rule exactly', async () => {
      const a = tensor([[2, 3]], { requiresGrad: true }); // [1,2]
      const b = tensor([[4], [5]], { requiresGrad: true }); // [2,1]
      const y = matmul(a, b); // [1,1] = 2*4 + 3*5 = 23

      const yOut = await run(y, { engine });
      expect(await yOut.equals(tensor([[23]]))).toBe(true);

      // d/da(a @ b) = b.T → shape [1,2] = [[4, 5]]
      // d/db(a @ b) = a.T → shape [2,1] = [[2], [3]]
      y.backward();
      const [gradA, gradB] = await run([a.grad!, b.grad!], { engine });
      expect(await gradA.equals(tensor([[4, 5]]))).toBe(true);
      expect(await gradB.equals(tensor([[2], [3]]))).toBe(true);
    });
  });
});
