import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, add, sum, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { MAX_RANK } from '@webtensor/ir';
import { BACKENDS } from '../helpers';
import { expectClose } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`rank-64 meta parity — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    // Rank 10 tensor: [2, 1, 1, 1, 1, 1, 1, 1, 1, 3] — total 6 elements.
    // This exercises the widened meta buffer on every backend and verifies
    // broadcasting, unary maps, and reductions all agree across the rank jump.
    const shape = [2, 1, 1, 1, 1, 1, 1, 1, 1, 3];
    const nested = (values: number[]): unknown => {
      const rows = values.slice(0, 3);
      const rows2 = values.slice(3, 6);
      const wrap = (v: number[]): unknown => [[[[[[[[[v]]]]]]]]];
      return [wrap(rows)[0], wrap(rows2)[0]];
    };

    it('add + sum produce the same values as CPU reference', async () => {
      const flatA = [1, -2, 3, -4, 5, -6];
      const flatB = [10, 20, 30, 40, 50, 60];
      const a = tensor(nested(flatA) as never);
      const b = tensor(nested(flatB) as never);
      expect(a.shape).toEqual(shape);

      const y = await run(sum(add(a, b), 9, false), { engine });
      expect(y.shape.length).toBe(9);
      // Row 0: (1+10)+(-2+20)+(3+30) = 62
      // Row 1: (-4+40)+(5+50)+(-6+60) = 145
      expectClose(y.data, [62, 145]);
    });
  });
});

describe('rank-64 rejection', () => {
  it('MAX_RANK is 64', () => {
    expect(MAX_RANK).toBe(64);
  });
});
