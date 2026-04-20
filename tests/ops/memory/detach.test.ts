import { describe, it, expect } from 'vitest';
import { tensor, detach, randn } from '@webtensor/core';

describe('detach', () => {
  it('preserves shape and dtype', () => {
    const a = tensor([1, 2, 3], { dtype: 'int32' });
    const d = detach(a);
    expect(d.shape).toEqual(a.shape);
    expect(d.dtype).toBe('int32');
  });

  it('clears requiresGrad', () => {
    const a = randn([4], { requiresGrad: true });
    expect(a.requiresGrad).toBe(true);
    expect(detach(a).requiresGrad).toBe(false);
  });

  it('breaks the gradient chain (no _ctx)', () => {
    const a = randn([4], { requiresGrad: true });
    const d = detach(a);
    expect(d._ctx).toBeUndefined();
  });
});
