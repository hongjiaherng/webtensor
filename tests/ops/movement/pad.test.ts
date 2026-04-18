import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, slice, pad, mul, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

// Pad and slice's backward share the scatter-into-constant kernel.

BACKENDS.forEach(({ name, create }) => {
  describe(`pad (forward) — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('pads 1D with zeros (before + after)', async () => {
      const a = tensor([7.0, 8.0, 9.0]);
      // pads = [before_0, after_0] = [2, 1]
      const y = await run(pad(a, [2, 1]), { engine });
      expect(y.shape).toEqual([6]);
      expect(await y.equals(tensor([0, 0, 7, 8, 9, 0]))).toBe(true);
    });

    it('pads 2D with zeros around the interior', async () => {
      const a = tensor([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      // pads = [before_0, before_1, after_0, after_1] = [1, 1, 1, 1]
      const y = await run(pad(a, [1, 1, 1, 1]), { engine });
      expect(y.shape).toEqual([4, 4]);
      expect(
        await y.equals(
          tensor([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
          ]),
        ),
      ).toBe(true);
    });

    it('asymmetric pads', async () => {
      const a = tensor([1.0, 2.0]);
      const y = await run(pad(a, [3, 1]), { engine });
      expect(await y.equals(tensor([0, 0, 0, 1, 2, 0]))).toBe(true);
    });

    it('non-zero constant fill', async () => {
      const a = tensor([1.0, 2.0]);
      const y = await run(pad(a, [1, 2], -1), { engine });
      expect(await y.equals(tensor([-1, 1, 2, -1, -1]))).toBe(true);
    });

    it('preserves int32 dtype and fill value', async () => {
      const a = tensor([1, 2, 3], { dtype: 'int32' });
      const y = await run(pad(a, [1, 1], 99), { engine });
      expect(y.dtype).toBe('int32');
      expect(await y.equals(tensor([99, 1, 2, 3, 99], { dtype: 'int32' }))).toBe(true);
    });

    it('handles strided input (post-transpose)', async () => {
      const a = tensor([
        [1.0, 2.0],
        [3.0, 4.0],
      ]).transpose();
      const y = await run(pad(a, [1, 0, 0, 1]), { engine });
      expect(
        await y.equals(
          tensor([
            [0, 0, 0],
            [1, 3, 0],
            [2, 4, 0],
          ]),
        ),
      ).toBe(true);
    });

    it('zero pads on all sides is identity', async () => {
      const a = tensor([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const y = await run(pad(a, [0, 0, 0, 0]), { engine });
      expect(await y.equals(a)).toBe(true);
    });
  });

  describe(`pad (backward) — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('gradient of pad equals slice of upstream grad', async () => {
      const a = tensor([1.0, 2.0, 3.0], { requiresGrad: true });
      const y = pad(a, [1, 1]); // shape [5]: [0, 1, 2, 3, 0]
      // weight grid picks the middle 3 values → input grad = [10, 20, 30]
      mul(y, tensor([100, 10, 20, 30, 100]))
        .sum()
        .backward();
      const g = await run(a.grad!, { engine });
      expect(await g.equals(tensor([10, 20, 30]))).toBe(true);
    });
  });

  describe(`slice backward (via pad) — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('1D slice: grad zero outside the region', async () => {
      const a = tensor([1.0, 2.0, 3.0, 4.0, 5.0], { requiresGrad: true });
      slice(a, [1], [4]).sum().backward();
      const g = await run(a.grad!, { engine });
      expect(await g.equals(tensor([0, 1, 1, 1, 0]))).toBe(true);
    });

    it('2D slice: grad zero outside the region', async () => {
      const a = tensor(
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0],
        ],
        { requiresGrad: true },
      );
      slice(a, [0, 1], [2, 3]).sum().backward();
      const g = await run(a.grad!, { engine });
      expect(
        await g.equals(
          tensor([
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
          ]),
        ),
      ).toBe(true);
    });

    it('weighted slice backward', async () => {
      const a = tensor([1.0, 2.0, 3.0, 4.0, 5.0], { requiresGrad: true });
      mul(slice(a, [1], [4]), tensor([10, 20, 30]))
        .sum()
        .backward();
      const g = await run(a.grad!, { engine });
      expect(await g.equals(tensor([0, 10, 20, 30, 0]))).toBe(true);
    });

    it('nested slice composes in backward', async () => {
      const a = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], { requiresGrad: true });
      // Outer [1..6] = [2,3,4,5,6], inner [1..4] of that = [3,4,5].
      // Grad at a: 1 inside indices [2,3,4], 0 elsewhere.
      slice(slice(a, [1], [6]), [1], [4])
        .sum()
        .backward();
      const g = await run(a.grad!, { engine });
      expect(await g.equals(tensor([0, 0, 1, 1, 1, 0, 0]))).toBe(true);
    });
  });
});

describe('pad — input validation', () => {
  it('rejects pads with wrong length', () => {
    const a = tensor([1, 2, 3]);
    expect(() => pad(a, [1, 1, 1])).toThrow(/2 \* rank/);
  });

  it('rejects negative pads', () => {
    const a = tensor([1, 2, 3]);
    expect(() => pad(a, [1, -1])).toThrow(/negative pad/);
  });
});
