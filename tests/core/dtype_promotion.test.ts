import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, add, sub, mul, div, neg, abs, matmul, sum, mean, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

// Dtype support matrix + mixed-dtype promotion rules.
//
// Binary arithmetic: int32 works on all three backends (kernel factories
// accept int32 via dtype-polymorphic buffers / SCALAR template substitution).
// Unary / matmul / reductions: int32 currently lives on CPU only — WASM and
// WebGPU int32 kernels for those ops are not yet implemented.
// Mixed-dtype ops (e.g. add(float32, int32)) follow PyTorch promotion rules
// and are validated on CPU where both dtypes are first-class.

const ARITH_BACKENDS = BACKENDS;
const CPU_ONLY = BACKENDS.filter((b) => b.name === 'CPU');

ARITH_BACKENDS.forEach(({ name, create }) => {
  describe(`int32 binary arithmetic — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('int32 + int32 stays int32', async () => {
      const a = tensor([1, 2, 3], { dtype: 'int32' });
      const b = tensor([10, 20, 30], { dtype: 'int32' });
      const y = await run(add(a, b), { engine });
      expect(y.dtype).toBe('int32');
      expect(await y.equals(tensor([11, 22, 33], { dtype: 'int32' }))).toBe(true);
    });

    // Mixed-dtype arithmetic requires an implicit Cast, arriving in Phase 1e.
    // For now, CPU does it correctly via JS's numeric coercion at read/write time.
    if (name === 'CPU') {
      it('float32 + int32 promotes to float32 (CPU only, pending Phase 1e Cast)', async () => {
        const a = tensor([1.5, 2.5], { dtype: 'float32' });
        const b = tensor([1, 2], { dtype: 'int32' });
        const y = await run(add(a, b), { engine });
        expect(y.dtype).toBe('float32');
        expect(await y.equals(tensor([2.5, 4.5]))).toBe(true);
      });
    }

    it('int32 sub / mul round-trip', async () => {
      const a = tensor([10, 20, 30], { dtype: 'int32' });
      const b = tensor([1, 2, 3], { dtype: 'int32' });
      const s = await run(sub(a, b), { engine });
      const m = await run(mul(a, b), { engine });
      expect(s.dtype).toBe('int32');
      expect(m.dtype).toBe('int32');
      expect(await s.equals(tensor([9, 18, 27], { dtype: 'int32' }))).toBe(true);
      expect(await m.equals(tensor([10, 40, 90], { dtype: 'int32' }))).toBe(true);
    });

    it('int32 div truncates toward zero', async () => {
      const a = tensor([7, -7, 5], { dtype: 'int32' });
      const b = tensor([2, 2, 3], { dtype: 'int32' });
      const y = await run(div(a, b), { engine });
      expect(y.dtype).toBe('int32');
      // Both JS typed-array store semantics and Rust i32 `/` truncate toward zero.
      expect(await y.equals(tensor([3, -3, 1], { dtype: 'int32' }))).toBe(true);
    });

    it('rejects bool arithmetic with a clear error', () => {
      const a = tensor([1, 0, 1], { dtype: 'bool' });
      const b = tensor([0, 1, 1], { dtype: 'bool' });
      expect(() => add(a, b)).toThrow(/not supported for arithmetic/);
    });
  });
});

CPU_ONLY.forEach(({ name, create }) => {
  describe(`int32 other ops (CPU-only for now) — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('unary neg/abs preserve int32 dtype', async () => {
      const a = tensor([-3, -1, 0, 2, 5], { dtype: 'int32' });
      const n = await run(neg(a), { engine });
      const b = await run(abs(a), { engine });
      expect(n.dtype).toBe('int32');
      expect(b.dtype).toBe('int32');
      expect(await n.equals(tensor([3, 1, 0, -2, -5], { dtype: 'int32' }))).toBe(true);
      expect(await b.equals(tensor([3, 1, 0, 2, 5], { dtype: 'int32' }))).toBe(true);
    });

    it('matmul works on int32 inputs', async () => {
      const a = tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: 'int32' },
      );
      const b = tensor(
        [
          [5, 6],
          [7, 8],
        ],
        { dtype: 'int32' },
      );
      const y = await run(matmul(a, b), { engine });
      expect(y.dtype).toBe('int32');
      expect(
        await y.equals(
          tensor(
            [
              [19, 22],
              [43, 50],
            ],
            { dtype: 'int32' },
          ),
        ),
      ).toBe(true);
    });

    it('sum on int32 stays int32', async () => {
      const a = tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: 'int32' },
      );
      const y = await run(sum(a), { engine });
      expect(y.dtype).toBe('int32');
      expect(await y.equals(tensor(21, { dtype: 'int32' }))).toBe(true);
    });

    it('mean on int32 (currently stays int32)', async () => {
      // TODO: follow PyTorch and promote mean(int) → float.
      const a = tensor([2, 4, 6], { dtype: 'int32' });
      const y = await run(mean(a), { engine });
      expect(y.dtype).toBe('int32');
      expect(await y.equals(tensor(4, { dtype: 'int32' }))).toBe(true);
    });
  });
});
