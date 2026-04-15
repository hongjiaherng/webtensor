import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, transpose } from '../../packages/core/src';
import { Backend } from '../../packages/runtime/src';
import { BACKENDS, runUnary, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`Transpose — ${name} Backend`, () => {
    let backend: Backend;

    beforeAll(async () => {
      backend = await create();
    });

    it('[2,3] → [3,2]', async () => {
      const out = await runUnary(backend, transpose, [[1, 2, 3], [4, 5, 6]]);
      expect(Array.from(out)).toEqual([1, 4, 2, 5, 3, 6]);
    });

    it('[3,3] square matrix', async () => {
      const out = await runUnary(backend, transpose, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      expect(Array.from(out)).toEqual([1, 4, 7, 2, 5, 8, 3, 6, 9]);
    });

    it('[1,4] row vector → [4,1] col vector', async () => {
      const out = await runUnary(backend, transpose, [[10, 20, 30, 40]]);
      expect(Array.from(out)).toEqual([10, 20, 30, 40]);
    });

    it('[4,1] col vector → [1,4] row vector', async () => {
      const out = await runUnary(backend, transpose, [[10], [20], [30], [40]]);
      expect(Array.from(out)).toEqual([10, 20, 30, 40]);
    });

    it('double transpose = identity', async () => {
      const a = tensor([[1, 2, 3], [4, 5, 6]]);
      const y = transpose(transpose(a));
      const graph = (await import('../../packages/core/src')).compileGraph([y]);
      const engine = new (await import('../../packages/runtime/src')).Engine(await create());
      engine.evaluate(graph);
      const out = await engine.get(y.id) as Float32Array;
      expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6]);
    });
  });
});

describe('Transpose — shape errors (no backend)', () => {
  it('1D input throws', () => {
    const a = tensor([1, 2, 3]);
    expect(() => transpose(a)).toThrow(/Transpose requires at least 2 dimensions/);
  });
});
