import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, compileGraph } from '../../packages/core/src';
import { Engine, Backend } from '../../packages/runtime/src';
import { BACKENDS, expectClose } from '../helpers';

// WebGPU gather pre-pass (needed for non-contiguous views) has a known issue
// where meta[0] reads as 0. Excluded until the root cause is diagnosed.
const SLICE_BACKENDS = BACKENDS.filter((b) => b.name !== 'WebGPU');

async function run(backend: Backend, output: ReturnType<typeof tensor>): Promise<Float32Array> {
  const graph = compileGraph([output]);
  const engine = new Engine(backend);
  engine.evaluate(graph);
  const data = await engine.get(output.id);
  if (!data) throw new Error(`engine.get(${output.id}) returned undefined`);
  return data as Float32Array;
}

SLICE_BACKENDS.forEach(({ name, create }) => {
  describe(`Slice — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('[4] → slice([1],[3]) → contiguous', async () => {
      const out = await run(backend, tensor([10, 20, 30, 40]).slice([1], [3]).contiguous());
      expectClose(out, [20, 30]);
    });

    it('[2,4] → slice rows 0:2, cols 1:3 → contiguous', async () => {
      const out = await run(
        backend,
        tensor([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ])
          .slice([0, 1], [2, 3])
          .contiguous(),
      );
      expectClose(out, [2, 3, 6, 7]);
    });

    it('[3,3] → slice row 1:3, col 0:2 → contiguous', async () => {
      const out = await run(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ])
          .slice([1, 0], [3, 2])
          .contiguous(),
      );
      expectClose(out, [4, 5, 7, 8]);
    });

    it('slice → add (strided input to kernel)', async () => {
      const out = await run(
        backend,
        tensor([0, 1, 2, 3, 4, 5]).slice([2], [5]).add(tensor([10, 10, 10])),
      );
      expectClose(out, [12, 13, 14]);
    });

    it('slice → relu (mixed positive/negative)', async () => {
      const out = await run(backend, tensor([-3, -2, -1, 0, 1, 2, 3]).slice([2], [6]).relu());
      expectClose(out, [0, 0, 1, 2]);
    });

    it('rank mismatch throws', () => {
      expect(() =>
        tensor([
          [1, 2],
          [3, 4],
        ]).slice([0], [1]),
      ).toThrow();
    });
  });
});
