import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS, expectClose } from '../helpers';

// ---------------------------------------------------------------------------
// Unsqueeze
// ---------------------------------------------------------------------------

BACKENDS.forEach(({ name, create }) => {
  describe(`Unsqueeze — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[3] → unsqueeze(0) → [1,3]', async () => {
      const t = tensor([1, 2, 3]);
      expect(t.unsqueeze(0).shape).toEqual([1, 3]);
      const out = (await run(t.unsqueeze(0).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3]);
    });

    it('[3] → unsqueeze(1) → [3,1]', async () => {
      const t = tensor([1, 2, 3]);
      expect(t.unsqueeze(1).shape).toEqual([3, 1]);
      const out = (await run(t.unsqueeze(1).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3]);
    });

    it('[2,3] → unsqueeze(0) → [1,2,3]', async () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(t.unsqueeze(0).shape).toEqual([1, 2, 3]);
      const out = (await run(t.unsqueeze(0).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3, 4, 5, 6]);
    });

    it('[2,3] → unsqueeze(1) → [2,1,3]', async () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(t.unsqueeze(1).shape).toEqual([2, 1, 3]);
      const out = (await run(t.unsqueeze(1).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3, 4, 5, 6]);
    });

    it('[2,3] → unsqueeze(2) → [2,3,1]', async () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(t.unsqueeze(2).shape).toEqual([2, 3, 1]);
      const out = (await run(t.unsqueeze(2).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3, 4, 5, 6]);
    });
  });
});

// ---------------------------------------------------------------------------
// Squeeze
// ---------------------------------------------------------------------------

BACKENDS.forEach(({ name, create }) => {
  describe(`Squeeze — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[1,3] → squeeze(0) → [3]', async () => {
      const t = tensor([[1, 2, 3]]);
      expect(t.squeeze(0).shape).toEqual([3]);
      const out = (await run(t.squeeze(0).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3]);
    });

    it('[3,1] → squeeze(1) → [3]', async () => {
      const t = tensor([[1], [2], [3]]);
      expect(t.squeeze(1).shape).toEqual([3]);
      const out = (await run(t.squeeze(1).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3]);
    });

    it('[1,3,1] → squeeze() removes all size-1 dims → [3]', async () => {
      const t = tensor([[[1], [2], [3]]]);
      expect(t.squeeze().shape).toEqual([3]);
      const out = (await run(t.squeeze().contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3]);
    });

    it('[2,3] → squeeze(0) throws (dim 0 is not size 1)', () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(() => t.squeeze(0)).toThrow(/size 2, expected 1/);
    });

    it('unsqueeze then squeeze = identity', async () => {
      const t = tensor([1, 2, 3, 4]);
      const out = (await run(t.unsqueeze(0).squeeze(0).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3, 4]);
    });
  });
});

// ---------------------------------------------------------------------------
// Permute
// ---------------------------------------------------------------------------

BACKENDS.forEach(({ name, create }) => {
  describe(`Permute — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[2,3] → permute([1,0]) = transpose', async () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const out = (await run(t.permute([1, 0]).contiguous(), { engine })).data!;
      // Transposed: [[1,4],[2,5],[3,6]] → flat [1,4,2,5,3,6]
      expectClose(out, [1, 4, 2, 5, 3, 6]);
    });

    it('[2,3,4] → permute([2,0,1]) reorders dims', async () => {
      // shape [2,3,4] → permute [2,0,1] → shape [4,2,3]
      const t = tensor([
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ],
        [
          [13, 14, 15, 16],
          [17, 18, 19, 20],
          [21, 22, 23, 24],
        ],
      ]);
      expect(t.permute([2, 0, 1]).shape).toEqual([4, 2, 3]);
      const out = (await run(t.permute([2, 0, 1]).contiguous(), { engine })).data!;
      // permute([2,0,1]): out[k][i][j] = in[i][j][k]
      // k=0: [[1,5,9],[13,17,21]]
      // k=1: [[2,6,10],[14,18,22]]
      // k=2: [[3,7,11],[15,19,23]]
      // k=3: [[4,8,12],[16,20,24]]
      expectClose(
        out,
        [1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24],
      );
    });

    it('permute identity [0,1,2] = no-op', async () => {
      const t = tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const out = (await run(t.permute([0, 1, 2]).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3, 4, 5, 6, 7, 8]);
    });
  });
});

// ---------------------------------------------------------------------------
// Expand
// ---------------------------------------------------------------------------

BACKENDS.forEach(({ name, create }) => {
  describe(`Expand — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[1,3] → expand([2,3]) broadcasts row', async () => {
      const t = tensor([[1, 2, 3]]);
      expect(t.expand([2, 3]).shape).toEqual([2, 3]);
      const out = (await run(t.expand([2, 3]).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3, 1, 2, 3]);
    });

    it('[3,1] → expand([3,4]) broadcasts column', async () => {
      const t = tensor([[1], [2], [3]]);
      expect(t.expand([3, 4]).shape).toEqual([3, 4]);
      const out = (await run(t.expand([3, 4]).contiguous(), { engine })).data!;
      expectClose(out, [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    });

    it('[1] → expand([5]) broadcasts scalar-like', async () => {
      const t = tensor([7]);
      const out = (await run(t.expand([5]).contiguous(), { engine })).data!;
      expectClose(out, [7, 7, 7, 7, 7]);
    });

    it('expand then add uses broadcast strides', async () => {
      const a = tensor([[1, 2, 3]]);
      const b = tensor([[10], [20], [30]]);
      // a expanded to [3,3], b expanded to [3,3], add them
      const out = (await run(a.expand([3, 3]).add(b.expand([3, 3])), { engine })).data!;
      expectClose(out, [11, 12, 13, 21, 22, 23, 31, 32, 33]);
    });
  });
});

// ---------------------------------------------------------------------------
// View (strict reshape — must be contiguous)
// ---------------------------------------------------------------------------

BACKENDS.forEach(({ name, create }) => {
  describe(`View — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[2,3] → view([6]) on contiguous tensor', async () => {
      const out = (
        await run(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ])
            .view([6])
            .contiguous(),
          { engine },
        )
      ).data!;
      expectClose(out, [1, 2, 3, 4, 5, 6]);
    });

    it('[6] → view([2,3]) on contiguous tensor', async () => {
      const out = (await run(tensor([1, 2, 3, 4, 5, 6]).view([2, 3]).contiguous(), { engine }))
        .data!;
      expectClose(out, [1, 2, 3, 4, 5, 6]);
    });

    it('view on non-contiguous tensor throws', async () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
        .transpose()
        .view([6]);
      await expect(run(t, { engine })).rejects.toThrow(/contiguous/i);
    });
  });
});

// ---------------------------------------------------------------------------
// Flatten
// ---------------------------------------------------------------------------

BACKENDS.forEach(({ name, create }) => {
  describe(`Flatten — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[2,3,4] → flatten() → [24]', async () => {
      const data = [];
      for (let i = 0; i < 24; i++) data.push(i + 1);
      const t = tensor(data).reshape([2, 3, 4]);
      expect(t.flatten().shape).toEqual([24]);
      const out = (await run(t.flatten().contiguous(), { engine })).data!;
      expectClose(out, data);
    });

    it('[2,3,4] → flatten(1) → [2,12]', async () => {
      const data = [];
      for (let i = 0; i < 24; i++) data.push(i + 1);
      const t = tensor(data).reshape([2, 3, 4]);
      expect(t.flatten(1).shape).toEqual([2, 12]);
    });

    it('[2,3,4] → flatten(0,1) → [6,4]', async () => {
      const data = [];
      for (let i = 0; i < 24; i++) data.push(i + 1);
      const t = tensor(data).reshape([2, 3, 4]);
      expect(t.flatten(0, 1).shape).toEqual([6, 4]);
    });
  });
});

// ---------------------------------------------------------------------------
// Chained view ops
// ---------------------------------------------------------------------------

BACKENDS.forEach(({ name, create }) => {
  describe(`Chained view ops — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('unsqueeze → expand → contiguous', async () => {
      // [3] → unsqueeze(0) → [1,3] → expand([4,3]) → [4,3]
      const t = tensor([1, 2, 3]);
      const out = (await run(t.unsqueeze(0).expand([4, 3]).contiguous(), { engine })).data!;
      expectClose(out, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]);
    });

    it('permute → reshape (auto-contiguous)', async () => {
      // [2,3] → permute([1,0]) → [3,2] (non-contiguous) → reshape([6]) auto-copies
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const out = (await run(t.permute([1, 0]).reshape([6]), { engine })).data!;
      expectClose(out, [1, 4, 2, 5, 3, 6]);
    });

    it('slice → unsqueeze → contiguous', async () => {
      // [[1,2,3],[4,5,6]] → slice row 1 → [4,5,6] shape [1,3] → unsqueeze(0) → [1,1,3]
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const sliced = t.slice([1, 0], [2, 3]);
      const out = (await run(sliced.unsqueeze(0).contiguous(), { engine })).data!;
      expectClose(out, [4, 5, 6]);
    });
  });
});
