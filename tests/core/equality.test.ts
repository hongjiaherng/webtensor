import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, equal, allclose, add, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`equal — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('equal shapes + equal values', async () => {
      expect(await equal(tensor([1, 2, 3]), tensor([1, 2, 3]), { engine })).toBe(true);
    });

    it('equal shapes + different values', async () => {
      expect(await equal(tensor([1, 2, 3]), tensor([1, 2, 4]), { engine })).toBe(false);
    });

    it('different shape → false (no dispatch)', async () => {
      expect(await equal(tensor([[1, 2, 3]]), tensor([1, 2, 3]), { engine })).toBe(false);
    });

    it('rank 2', async () => {
      expect(
        await equal(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          tensor([
            [1, 2],
            [3, 4],
          ]),
          { engine },
        ),
      ).toBe(true);
      expect(
        await equal(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          tensor([
            [1, 2],
            [3, 5],
          ]),
          { engine },
        ),
      ).toBe(false);
    });

    it('NaN never equals NaN', async () => {
      expect(await equal(tensor([NaN]), tensor([NaN]), { engine })).toBe(false);
    });

    it('as a method: .equals()', async () => {
      expect(await tensor([1, 2, 3]).equals(tensor([1, 2, 3]), { engine })).toBe(true);
      expect(await tensor([1, 2, 3]).equals(tensor([1, 2]), { engine })).toBe(false);
    });

    it('works on evaluated tensors from run()', async () => {
      const y = await run(add(tensor([1, 2, 3]), tensor([10, 20, 30])), { engine });
      expect(await y.equals(tensor([11, 22, 33]), { engine })).toBe(true);
    });

    it('works on unevaluated graph tensors (dispatches through backend)', async () => {
      const ungraphed = add(tensor([1, 2]), tensor([3, 4]));
      expect(await equal(ungraphed, tensor([4, 6]), { engine })).toBe(true);
    });
  });

  describe(`allclose — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('exactly equal values', async () => {
      expect(await allclose(tensor([1, 2, 3]), tensor([1, 2, 3]), { engine })).toBe(true);
    });

    it('close within default tolerance', async () => {
      expect(await allclose(tensor([1.0000001]), tensor([1.0]), { engine })).toBe(true);
    });

    it('far apart → false', async () => {
      expect(await allclose(tensor([1, 2, 3]), tensor([1, 2, 4]), { engine })).toBe(false);
    });

    it('rtol controls relative tolerance', async () => {
      expect(await allclose(tensor([1000]), tensor([1001]), { rtol: 1e-2, engine })).toBe(true);
      expect(await allclose(tensor([1000]), tensor([1001]), { rtol: 1e-4, engine })).toBe(false);
    });

    it('atol controls absolute tolerance', async () => {
      expect(await allclose(tensor([0]), tensor([0.001]), { atol: 0.01, engine })).toBe(true);
      expect(await allclose(tensor([0]), tensor([0.001]), { atol: 1e-6, engine })).toBe(false);
    });

    it('different shapes → false (no dispatch)', async () => {
      expect(await allclose(tensor([[1]]), tensor([1]), { engine })).toBe(false);
    });

    it('NaN ≠ NaN by default', async () => {
      expect(await allclose(tensor([NaN]), tensor([NaN]), { engine })).toBe(false);
    });

    it('NaN = NaN when equalNan: true', async () => {
      expect(await allclose(tensor([NaN]), tensor([NaN]), { equalNan: true, engine })).toBe(true);
    });

    it('same-signed infinity compares equal', async () => {
      expect(await allclose(tensor([Infinity]), tensor([Infinity]), { engine })).toBe(true);
      expect(await allclose(tensor([-Infinity]), tensor([-Infinity]), { engine })).toBe(true);
    });

    it('opposite-signed infinities are not close', async () => {
      expect(await allclose(tensor([Infinity]), tensor([-Infinity]), { engine })).toBe(false);
    });

    it('finite vs infinite is not close', async () => {
      expect(await allclose(tensor([1e30]), tensor([Infinity]), { engine })).toBe(false);
    });

    it('as a method: .allclose()', async () => {
      expect(await tensor([1.0, 2.0]).allclose(tensor([1.0000001, 2.0000001]), { engine })).toBe(
        true,
      );
    });

    it('works after run() with small numeric drift', async () => {
      const y = await run(add(tensor([0.1, 0.2]), tensor([0.2, 0.1])), { engine });
      expect(await y.allclose(tensor([0.3, 0.3]), { engine })).toBe(true);
    });

    it('Tensor.run() method evaluates in place of module-level run()', async () => {
      const y = await add(tensor([1, 2, 3]), tensor([10, 20, 30])).run({ engine });
      expect(await y.equals(tensor([11, 22, 33]), { engine })).toBe(true);
    });
  });
});
