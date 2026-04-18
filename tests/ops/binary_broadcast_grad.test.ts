import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, add, mul, sum, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

// Verifies that broadcasting in forward ops produces correctly-shaped gradients
// (via the internal unbroadcastGrad helper). Uses `sum(y)` to seed a scalar grad.

BACKENDS.forEach(({ name, create }) => {
  describe(`binary op broadcast gradients — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('add: [3,1] + [1,4] → grads [3,1] and [1,4]', async () => {
      const a = tensor([[1], [2], [3]], { requiresGrad: true }); // [3,1]
      const b = tensor([[10, 20, 30, 40]], { requiresGrad: true }); // [1,4]
      const loss = sum(add(a, b));
      loss.backward();

      const aGrad = await run(a.grad!, { engine });
      const bGrad = await run(b.grad!, { engine });
      // Each element of a appears in 4 outputs; each element of b in 3
      expect(aGrad.equals(tensor([[4], [4], [4]]))).toBe(true);
      expect(bGrad.equals(tensor([[3, 3, 3, 3]]))).toBe(true);
    });

    it('mul: [3,1] * [1,4] → grads are weighted sums', async () => {
      const a = tensor([[1], [2], [3]], { requiresGrad: true });
      const b = tensor([[10, 20, 30, 40]], { requiresGrad: true });
      const loss = sum(mul(a, b));
      loss.backward();

      const aGrad = await run(a.grad!, { engine });
      const bGrad = await run(b.grad!, { engine });
      // dL/dA[i,0] = sum_j B[0,j] = 100
      expect(aGrad.equals(tensor([[100], [100], [100]]))).toBe(true);
      // dL/dB[0,j] = sum_i A[i,0] = 6
      expect(bGrad.equals(tensor([[6, 6, 6, 6]]))).toBe(true);
    });

    it('scalar + vector: [1] + [5] → scalar grad aggregates', async () => {
      const a = tensor([3], { requiresGrad: true });
      const b = tensor([1, 2, 3, 4, 5], { requiresGrad: true });
      const loss = sum(add(a, b));
      loss.backward();

      const aGrad = await run(a.grad!, { engine });
      const bGrad = await run(b.grad!, { engine });
      expect(aGrad.equals(tensor([5]))).toBe(true); // a appears 5 times
      expect(bGrad.equals(tensor([1, 1, 1, 1, 1]))).toBe(true);
    });

    it('rank-3 broadcast [2,3] + [4,1,3] → correct grad shapes', async () => {
      const a = tensor([[1, 2, 3], [4, 5, 6]], { requiresGrad: true }); // [2,3]
      const b = tensor([[[1, 1, 1]], [[2, 2, 2]], [[3, 3, 3]], [[4, 4, 4]]], {
        requiresGrad: true,
      }); // [4,1,3]
      const loss = sum(add(a, b)); // output shape [4,2,3]
      loss.backward();

      const aGrad = await run(a.grad!, { engine });
      const bGrad = await run(b.grad!, { engine });
      // a appears 4 times across the leading batch dim
      expect(aGrad.equals(tensor([[4, 4, 4], [4, 4, 4]]))).toBe(true);
      // each slice of b appears 2 times across the middle dim
      expect(
        bGrad.equals(tensor([[[2, 2, 2]], [[2, 2, 2]], [[2, 2, 2]], [[2, 2, 2]]])),
      ).toBe(true);
    });
  });
});
