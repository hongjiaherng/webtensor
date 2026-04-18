import { describe, it, expect, beforeAll } from 'vitest';
import {
  tensor,
  zeros,
  ones,
  rand,
  randn,
  zerosLike,
  onesLike,
  randnLike,
  run,
} from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`tensor factories — ${name}`, () => {
    let engine: Engine;

    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('zeros produces all-zero tensor', async () => {
      const y = await run(zeros([2, 3]), { engine });
      expect(
        y.equals(
          tensor([
            [0, 0, 0],
            [0, 0, 0],
          ]),
        ),
      ).toBe(true);
    });

    it('ones produces all-one tensor', async () => {
      const y = await run(ones([2, 3]), { engine });
      expect(
        y.equals(
          tensor([
            [1, 1, 1],
            [1, 1, 1],
          ]),
        ),
      ).toBe(true);
    });

    it('rand with seed is deterministic', async () => {
      const a = await run(rand([6], { seed: 42 }), { engine });
      const b = await run(rand([6], { seed: 42 }), { engine });
      expect(a.equals(b)).toBe(true);
      for (const v of Array.from(a.data!)) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
      }
    });

    it('randn has mean ≈ 0 and std ≈ 1', async () => {
      const y = await run(randn([10000], { seed: 7 }), { engine });
      const arr = Array.from(y.data!);
      const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
      const variance = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length;
      expect(Math.abs(mean)).toBeLessThan(0.05);
      expect(Math.abs(Math.sqrt(variance) - 1)).toBeLessThan(0.05);
    });

    it('zerosLike matches shape + dtype of source', async () => {
      const src = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const z = zerosLike(src);
      expect(z.shape).toEqual([2, 3]);
      expect(z.dtype).toBe('float32');
      const y = await run(z, { engine });
      expect(
        y.equals(
          tensor([
            [0, 0, 0],
            [0, 0, 0],
          ]),
        ),
      ).toBe(true);
    });

    it('onesLike matches shape of source', async () => {
      const y = await run(onesLike(tensor([[1], [2], [3]])), { engine });
      expect(y.equals(tensor([[1], [1], [1]]))).toBe(true);
    });

    it('randnLike matches shape of source', () => {
      const src = tensor([
        [1, 2],
        [3, 4],
      ]);
      const r = randnLike(src, { seed: 1 });
      expect(r.shape).toEqual([2, 2]);
    });

    it('randn with std option scales the distribution', async () => {
      const y = await run(randn([10000], { seed: 3, std: 0.1 }), { engine });
      const arr = Array.from(y.data!);
      const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
      const variance = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length;
      expect(Math.abs(Math.sqrt(variance) - 0.1)).toBeLessThan(0.01);
    });

    it('rand with low/high bounds', async () => {
      const y = await run(rand([5000], { seed: 1, low: -2, high: 5 }), { engine });
      const arr = Array.from(y.data!);
      for (const v of arr) {
        expect(v).toBeGreaterThanOrEqual(-2);
        expect(v).toBeLessThan(5);
      }
    });
  });
});
