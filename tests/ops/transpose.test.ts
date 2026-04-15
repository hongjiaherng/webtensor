import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, compileGraph } from '../../packages/core/src';
import { Engine, Backend } from '../../packages/runtime/src';
import { BACKENDS, expectClose } from '../helpers';

// WebGPU gather pre-pass (needed for non-contiguous views) has a known issue
// where meta[0] reads as 0. Excluded until the root cause is diagnosed.
const TRANSPOSE_BACKENDS = BACKENDS.filter((b) => b.name !== 'WebGPU');

async function run(backend: Backend, output: ReturnType<typeof tensor>): Promise<Float32Array> {
  const graph = compileGraph([output]);
  const engine = new Engine(backend);
  engine.evaluate(graph);
  const data = await engine.get(output.id);
  if (!data) throw new Error(`engine.get(${output.id}) returned undefined`);
  return data as Float32Array;
}

TRANSPOSE_BACKENDS.forEach(({ name, create }) => {
  describe(`Transpose — ${name}`, () => {
    let backend: Backend;
    beforeAll(async () => {
      backend = await create();
    });

    it('[2,3] → transpose → [3,2]', async () => {
      const out = await run(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
          .transpose()
          .contiguous(),
      );
      expectClose(out, [1, 4, 2, 5, 3, 6]);
    });

    it('[3,3] square matrix', async () => {
      const out = await run(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ])
          .transpose()
          .contiguous(),
      );
      expectClose(out, [1, 4, 7, 2, 5, 8, 3, 6, 9]);
    });

    it('[1,4] row vector → [4,1] col vector', async () => {
      const out = await run(backend, tensor([[10, 20, 30, 40]]).transpose().contiguous());
      expectClose(out, [10, 20, 30, 40]);
    });

    it('[4,1] col vector → [1,4] row vector', async () => {
      const out = await run(
        backend,
        tensor([[10], [20], [30], [40]]).transpose().contiguous(),
      );
      expectClose(out, [10, 20, 30, 40]);
    });

    it('double transpose = identity', async () => {
      const out = await run(
        backend,
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
          .transpose()
          .transpose()
          .contiguous(),
      );
      expectClose(out, [1, 2, 3, 4, 5, 6]);
    });

    it('rank-3 swap last two dims [2,2,3] → [2,3,2]', async () => {
      const out = await run(
        backend,
        tensor([
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
          [
            [7, 8, 9],
            [10, 11, 12],
          ],
        ])
          .transpose()
          .contiguous(),
      );
      expectClose(out, [1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12]);
    });

    it('transpose → matmul (strided input to kernel)', async () => {
      // a.T [3,2] @ identity [2,2] → [3,2] unchanged
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = tensor([
        [1, 0],
        [0, 1],
      ]);
      const out = await run(backend, a.transpose().matmul(b));
      expectClose(out, [1, 4, 2, 5, 3, 6]);
    });

    it('1D input throws', () => {
      expect(() => tensor([1, 2, 3]).transpose()).toThrow();
    });
  });
});
