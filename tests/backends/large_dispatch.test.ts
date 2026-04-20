import { describe, it, expect, beforeAll } from 'vitest';
import { ones, add, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

// WebGPU defaults to maxComputeWorkgroupsPerDimension = 65535. With a 1D
// dispatch of `[ceil(size / 64), 1, 1]` that caps usable tensor size at
// ~4.19M elements. `dispatch1D()` in backend-webgpu/utils.ts splits workgroups
// across X/Y for larger tensors. This test uses a tensor just over that cap
// (2^22 = 4,194,304) to exercise the split path on WebGPU and make sure
// CPU/WASM handle the size too.

const SIZE = 1 << 22; // 4,194,304

BACKENDS.forEach(({ name, create }) => {
  describe(`Large dispatch (>4M elements) — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it(`add over ${SIZE} elements produces all 2s`, async () => {
      const a = ones([SIZE]);
      const b = ones([SIZE]);
      const y = await run(add(a, b), { engine });
      expect(y.shape).toEqual([SIZE]);
      const data = y.data as Float32Array;
      // Spot-check instead of scanning all 4M elements to keep the test fast.
      expect(data[0]).toBe(2);
      expect(data[SIZE - 1]).toBe(2);
      expect(data[SIZE >> 1]).toBe(2);
      expect(data[(SIZE >> 1) + 1]).toBe(2);
    });
  });
});
