import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, cast, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

// Cast is the cross-dtype conversion op. All three backends handle all 9
// (from, to) pairs across float32 / int32 / bool. On WebGPU, bool is stored
// as u32 on device (backend.ts translates at write/read).

BACKENDS.forEach(({ name, create }) => {
  describe(`cast (numeric ↔ numeric) — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('float32 → int32 truncates toward zero', async () => {
      const a = tensor([1.2, -1.7, 0.0, 3.9, -0.5]);
      const y = await run(cast(a, 'int32'), { engine });
      expect(y.dtype).toBe('int32');
      expect(y.equals(tensor([1, -1, 0, 3, 0], { dtype: 'int32' }))).toBe(true);
    });

    it('int32 → float32 widens exactly', async () => {
      const a = tensor([-5, 0, 7, 42], { dtype: 'int32' });
      const y = await run(cast(a, 'float32'), { engine });
      expect(y.dtype).toBe('float32');
      expect(y.equals(tensor([-5, 0, 7, 42]))).toBe(true);
    });

    it('same-dtype cast is a pure copy', async () => {
      const a = tensor([
        [1.5, 2.5],
        [3.5, 4.5],
      ]);
      const y = await run(cast(a, 'float32'), { engine });
      expect(y.dtype).toBe('float32');
      expect(y.equals(a)).toBe(true);
    });

    it('respects strided input (post-transpose)', async () => {
      // transpose returns a non-contiguous view; cast must read via strides.
      const a = tensor([
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
      ]);
      const y = await run(cast(a.transpose(), 'int32'), { engine });
      expect(y.dtype).toBe('int32');
      expect(
        y.equals(
          tensor(
            [
              [1, 4],
              [2, 5],
              [3, 6],
            ],
            { dtype: 'int32' },
          ),
        ),
      ).toBe(true);
    });
  });
});

BACKENDS.forEach(({ name, create }) => {
  describe(`cast (bool round-trips) — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('float32 → bool canonicalizes nonzero → 1', async () => {
      const a = tensor([0.0, 1.5, -0.0, 0.001]);
      const y = await run(cast(a, 'bool'), { engine });
      expect(y.dtype).toBe('bool');
      expect(y.equals(tensor([0, 1, 0, 1], { dtype: 'bool' }))).toBe(true);
    });

    it('int32 → bool canonicalizes nonzero → 1', async () => {
      const a = tensor([0, -1, 2, 0], { dtype: 'int32' });
      const y = await run(cast(a, 'bool'), { engine });
      expect(y.dtype).toBe('bool');
      expect(y.equals(tensor([0, 1, 1, 0], { dtype: 'bool' }))).toBe(true);
    });

    it('bool → float32 widens 0/1', async () => {
      const a = tensor([1, 0, 1, 1], { dtype: 'bool' });
      const y = await run(cast(a, 'float32'), { engine });
      expect(y.dtype).toBe('float32');
      expect(y.equals(tensor([1, 0, 1, 1]))).toBe(true);
    });

    it('bool → int32 widens 0/1', async () => {
      const a = tensor([1, 0, 0, 1], { dtype: 'bool' });
      const y = await run(cast(a, 'int32'), { engine });
      expect(y.dtype).toBe('int32');
      expect(y.equals(tensor([1, 0, 0, 1], { dtype: 'int32' }))).toBe(true);
    });
  });
});

describe('cast — user-facing properties', () => {
  it('produces a non-differentiable output', () => {
    const a = tensor([1.5, 2.5], { requiresGrad: true });
    const y = cast(a, 'int32');
    expect(y.requiresGrad).toBe(false);
  });

  it('preserves shape', () => {
    const a = tensor([
      [1.5, 2.5],
      [3.5, 4.5],
    ]);
    const y = cast(a, 'int32');
    expect(y.shape).toEqual([2, 2]);
    expect(y.dtype).toBe('int32');
  });

  it('enables Tensor.cast() / .to() method chaining', () => {
    const a = tensor([1.5, 2.5]);
    expect(a.cast('int32').dtype).toBe('int32');
    expect(a.to('int32').dtype).toBe('int32');
  });
});
