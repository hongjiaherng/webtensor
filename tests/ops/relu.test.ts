import { describe, it, expect, beforeAll } from 'vitest';
import { relu } from '../../packages/core/src';
import { Backend } from '../../packages/runtime/src';
import { BACKENDS, runUnary, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Relu — ${name} Backend`, () => {
    let backend: Backend;

    beforeAll(async () => {
      backend = await create();
    });

    it('mixed values', async () => {
      const out = await runUnary(backend, relu, [-2, -1, 0, 1, 2]);
      expect(Array.from(out)).toEqual([0, 0, 0, 1, 2]);
    });

    it('all positive (pass-through)', async () => {
      const out = await runUnary(backend, relu, [0.1, 1.5, 3.0, 100.0]);
      expectClose(out, [0.1, 1.5, 3.0, 100.0]);
    });

    it('all negative (zero out)', async () => {
      const out = await runUnary(backend, relu, [-0.1, -1.5, -100.0]);
      expect(Array.from(out)).toEqual([0, 0, 0]);
    });

    it('all zeros', async () => {
      const out = await runUnary(backend, relu, [0, 0, 0, 0]);
      expect(Array.from(out)).toEqual([0, 0, 0, 0]);
    });

    it('large tensor (1024 elements: 512 neg + 512 pos)', async () => {
      const data = [
        ...Array.from({ length: 512 }, () => -1.0),
        ...Array.from({ length: 512 }, () => 1.0),
      ];
      const out = await runUnary(backend, relu, data);
      expect(out.length).toBe(1024);
      expect(out.slice(0, 512).every((v) => v === 0)).toBe(true);
      expect(out.slice(512).every((v) => v === 1)).toBe(true);
    });

    it('boundary at zero', async () => {
      const out = await runUnary(backend, relu, [-0.0001, 0.0, 0.0001]);
      expectClose(out, [0, 0, 0.0001]);
    });
  });
});
