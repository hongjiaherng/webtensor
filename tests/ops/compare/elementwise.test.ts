import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, eq, ne, lt, le, gt, ge, isclose, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

// Element-wise comparison ops produce bool tensors. All three backends ship
// these ops. The scalar helpers `equal` / `allclose` live in compare.test.ts.

BACKENDS.forEach(({ name, create }) => {
  describe(`comparison ops — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('eq / ne on float32', async () => {
      const a = tensor([1.0, 2.0, 3.0, 4.0]);
      const b = tensor([1.0, 0.0, 3.0, 5.0]);
      const e = await run(eq(a, b), { engine });
      const n = await run(ne(a, b), { engine });
      expect(e.dtype).toBe('bool');
      expect(n.dtype).toBe('bool');
      expect(e.equals(tensor([1, 0, 1, 0], { dtype: 'bool' }))).toBe(true);
      expect(n.equals(tensor([0, 1, 0, 1], { dtype: 'bool' }))).toBe(true);
    });

    it('eq / ne on int32', async () => {
      const a = tensor([1, 2, 3, 4], { dtype: 'int32' });
      const b = tensor([1, 0, 3, 5], { dtype: 'int32' });
      const e = await run(eq(a, b), { engine });
      expect(e.dtype).toBe('bool');
      expect(e.equals(tensor([1, 0, 1, 0], { dtype: 'bool' }))).toBe(true);
    });

    it('lt / le / gt / ge on float32', async () => {
      const a = tensor([1.0, 2.0, 3.0]);
      const b = tensor([2.0, 2.0, 2.0]);
      const l = await run(lt(a, b), { engine });
      const le_ = await run(le(a, b), { engine });
      const g = await run(gt(a, b), { engine });
      const ge_ = await run(ge(a, b), { engine });
      expect(l.equals(tensor([1, 0, 0], { dtype: 'bool' }))).toBe(true);
      expect(le_.equals(tensor([1, 1, 0], { dtype: 'bool' }))).toBe(true);
      expect(g.equals(tensor([0, 0, 1], { dtype: 'bool' }))).toBe(true);
      expect(ge_.equals(tensor([0, 1, 1], { dtype: 'bool' }))).toBe(true);
    });

    it('lt / gt on int32', async () => {
      const a = tensor([-1, 0, 1, 2], { dtype: 'int32' });
      const b = tensor([0, 0, 0, 0], { dtype: 'int32' });
      const l = await run(lt(a, b), { engine });
      const g = await run(gt(a, b), { engine });
      expect(l.equals(tensor([1, 0, 0, 0], { dtype: 'bool' }))).toBe(true);
      expect(g.equals(tensor([0, 0, 1, 1], { dtype: 'bool' }))).toBe(true);
    });

    it('broadcasts like binary arithmetic', async () => {
      const a = tensor([[1.0], [2.0], [3.0]]);
      const b = tensor([[0.0, 1.0, 2.0, 3.0]]);
      const y = await run(lt(a, b), { engine });
      expect(y.shape).toEqual([3, 4]);
      expect(
        y.equals(
          tensor(
            [
              [0, 0, 1, 1],
              [0, 0, 0, 1],
              [0, 0, 0, 0],
            ],
            { dtype: 'bool' },
          ),
        ),
      ).toBe(true);
    });

    it('handles strided inputs (post-transpose)', async () => {
      const a = tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = tensor([
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0],
      ]);
      const y = await run(eq(a.transpose(), b), { engine });
      expect(
        y.equals(
          tensor(
            [
              [1, 1],
              [1, 1],
              [1, 1],
            ],
            { dtype: 'bool' },
          ),
        ),
      ).toBe(true);
    });

    it('isclose: tight and loose tolerances', async () => {
      const a = tensor([1.0, 2.0, 3.0, 1e10]);
      const b = tensor([1.0 + 1e-7, 2.0001, 3.0, 1e10 + 1]);
      const tight = await run(isclose(a, b), { engine });
      const loose = await run(isclose(a, b, { rtol: 1e-3, atol: 1e-3 }), { engine });
      expect(tight.dtype).toBe('bool');
      expect(tight.equals(tensor([1, 0, 1, 1], { dtype: 'bool' }))).toBe(true);
      expect(loose.equals(tensor([1, 1, 1, 1], { dtype: 'bool' }))).toBe(true);
    });

    it('isclose: NaN handling', async () => {
      const a = tensor([NaN, NaN, 1.0]);
      const b = tensor([NaN, 0.0, 1.0]);
      const def = await run(isclose(a, b), { engine });
      const eqNan = await run(isclose(a, b, { equalNan: true }), { engine });
      expect(def.equals(tensor([0, 0, 1], { dtype: 'bool' }))).toBe(true);
      expect(eqNan.equals(tensor([1, 0, 1], { dtype: 'bool' }))).toBe(true);
    });

    it('isclose: ±inf matches itself', async () => {
      const a = tensor([Infinity, -Infinity, Infinity]);
      const b = tensor([Infinity, -Infinity, -Infinity]);
      const y = await run(isclose(a, b), { engine });
      expect(y.equals(tensor([1, 1, 0], { dtype: 'bool' }))).toBe(true);
    });
  });
});

describe('comparison ops — input validation', () => {
  it('rejects mismatched dtypes with a clear error', () => {
    const a = tensor([1.0, 2.0]);
    const b = tensor([1, 2], { dtype: 'int32' });
    expect(() => eq(a, b)).toThrow(/dtypes .* differ/);
  });

  it('rejects bool inputs', () => {
    const a = tensor([1, 0], { dtype: 'bool' });
    const b = tensor([1, 1], { dtype: 'bool' });
    expect(() => eq(a, b)).toThrow(/bool is not a valid/);
  });

  it('isclose rejects non-float32 inputs', () => {
    const a = tensor([1, 2], { dtype: 'int32' });
    const b = tensor([1, 2], { dtype: 'int32' });
    expect(() => isclose(a, b)).toThrow(/requires float32/);
  });

  it('comparison output is non-differentiable', () => {
    const a = tensor([1.0, 2.0], { requiresGrad: true });
    const b = tensor([1.0, 3.0]);
    expect(eq(a, b).requiresGrad).toBe(false);
    expect(lt(a, b).requiresGrad).toBe(false);
  });

  it('enables Tensor method chaining', () => {
    const a = tensor([1.0, 2.0, 3.0]);
    const b = tensor([3.0, 2.0, 1.0]);
    expect(a.lt(b).dtype).toBe('bool');
    expect(a.ge(b).dtype).toBe('bool');
    expect(a.isclose(b, { atol: 0.5 }).dtype).toBe('bool');
  });
});
