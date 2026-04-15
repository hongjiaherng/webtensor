import { describe, it, expect, beforeAll } from 'vitest';
import { sub } from '../../packages/core/src';
import { Backend } from '../../packages/runtime/src';
import { BACKENDS, runBinary, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Sub — ${name} Backend`, () => {
    let backend: Backend;

    beforeAll(async () => {
      backend = await create();
    });

    it('basic subtraction', async () => {
      const out = await runBinary(backend, sub, [10, 20, 30], [1, 2, 3]);
      expect(Array.from(out)).toEqual([9, 18, 27]);
    });

    it('scalar broadcast [3] - [1]', async () => {
      const out = await runBinary(backend, sub, [10, 20, 30], [5]);
      expect(Array.from(out)).toEqual([5, 15, 25]);
    });

    it('negative result', async () => {
      const out = await runBinary(backend, sub, [1, 2, 3], [4, 5, 6]);
      expect(Array.from(out)).toEqual([-3, -3, -3]);
    });

    it('large tensor (1024 elements)', async () => {
      const a = Array.from({ length: 1024 }, () => 5.0);
      const b = Array.from({ length: 1024 }, () => 3.0);
      const out = await runBinary(backend, sub, a, b);
      expect(out.length).toBe(1024);
      expect(out.every((v) => v === 2.0)).toBe(true);
    });

    it('self-subtraction yields zeros', async () => {
      const out = await runBinary(backend, sub, [1.5, 2.5, 3.5], [1.5, 2.5, 3.5]);
      expectClose(out, [0, 0, 0]);
    });
  });
});
