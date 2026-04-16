import { describe, it, expect, beforeAll } from 'vitest';
import { div } from '@webtensor/core';
import { Backend } from '@webtensor/runtime';
import { BACKENDS, runBinary, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Div — ${name} Backend`, () => {
    let backend: Backend;

    beforeAll(async () => {
      backend = await create();
    });

    it('basic division', async () => {
      const out = await runBinary(backend, div, [10, 20, 9], [2, 4, 3]);
      expect(Array.from(out)).toEqual([5, 5, 3]);
    });

    it('scalar broadcast [3] / [1]', async () => {
      const out = await runBinary(backend, div, [10, 20, 30], [10]);
      expect(Array.from(out)).toEqual([1, 2, 3]);
    });

    it('fractional results', async () => {
      const out = await runBinary(backend, div, [1, 1, 1], [3, 4, 8]);
      expectClose(out, [1 / 3, 0.25, 0.125]);
    });

    it('large tensor (1024 elements)', async () => {
      const a = Array.from({ length: 1024 }, () => 6.0);
      const b = Array.from({ length: 1024 }, () => 2.0);
      const out = await runBinary(backend, div, a, b);
      expect(out.length).toBe(1024);
      expect(out.every((v) => v === 3.0)).toBe(true);
    });

    it('divide by one (identity)', async () => {
      const out = await runBinary(backend, div, [7.5, 3.2, 1.1], [1, 1, 1]);
      expectClose(out, [7.5, 3.2, 1.1]);
    });
  });
});
