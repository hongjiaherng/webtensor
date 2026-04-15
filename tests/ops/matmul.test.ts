import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, matmul } from '../../packages/core/src';
import { Backend } from '../../packages/runtime/src';
import { BACKENDS, runBinary, expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`MatMul — ${name} Backend`, () => {
    let backend: Backend;

    beforeAll(async () => {
      backend = await create();
    });

    it('[2,3] × [3,2] = [2,2]', async () => {
      const out = await runBinary(
        backend,
        matmul,
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [
          [7, 8],
          [9, 10],
          [11, 12],
        ],
      );
      expect(Array.from(out)).toEqual([58, 64, 139, 154]);
    });

    it('[2,2] × [2,2] square', async () => {
      const out = await runBinary(
        backend,
        matmul,
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      );
      expect(Array.from(out)).toEqual([19, 22, 43, 50]);
    });

    it('[1,1] × [1,1] scalar case', async () => {
      const out = await runBinary(backend, matmul, [[3]], [[7]]);
      expectClose(out, [21]);
    });

    it('[1,4] × [4,1] dot product', async () => {
      const out = await runBinary(backend, matmul, [[1, 2, 3, 4]], [[1], [2], [3], [4]]);
      expectClose(out, [30]);
    });

    it('large 32×32 (all ones, each output = 32)', async () => {
      const row = Array.from({ length: 32 }, () => 1.0);
      const A = Array.from({ length: 32 }, () => row);
      const B = Array.from({ length: 32 }, () => row);
      const out = await runBinary(backend, matmul, A, B);
      expect(out.length).toBe(1024);
      expectClose(
        out,
        Array.from({ length: 1024 }, () => 32.0),
        1e-3,
      );
    });
  });
});

describe('MatMul — shape errors (no backend)', () => {
  it('K-dimension mismatch throws', () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]); // [2,3], K=3
    const b = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]); // [4,2], K=4
    expect(() => matmul(a, b)).toThrow(/MatMul inner dimensions must match/);
  });

  it('1D input throws', () => {
    const a = tensor([1, 2, 3]); // shape [3]
    const b = tensor([[1], [2], [3]]);
    expect(() => matmul(a, b)).toThrow(/MatMul requires inputs to be at least 2/);
  });
});
