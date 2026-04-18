import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, matmul, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`matmul — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('[2,3] × [3,2] = [2,2]', async () => {
      const y = await run(
        matmul(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          tensor([
            [7, 8],
            [9, 10],
            [11, 12],
          ]),
        ),
        { engine },
      );
      expect(
        await y.equals(
          tensor([
            [58, 64],
            [139, 154],
          ]),
        ),
      ).toBe(true);
    });

    it('[2,2] × [2,2] square', async () => {
      const y = await run(
        matmul(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          tensor([
            [5, 6],
            [7, 8],
          ]),
        ),
        { engine },
      );
      expect(
        await y.equals(
          tensor([
            [19, 22],
            [43, 50],
          ]),
        ),
      ).toBe(true);
    });

    it('[1,1] × [1,1] scalar case', async () => {
      const y = await run(matmul(tensor([[3]]), tensor([[7]])), { engine });
      expect(await y.allclose(tensor([[21]]))).toBe(true);
    });

    it('[1,4] × [4,1] dot product', async () => {
      const y = await run(matmul(tensor([[1, 2, 3, 4]]), tensor([[1], [2], [3], [4]])), { engine });
      expect(await y.allclose(tensor([[30]]))).toBe(true);
    });

    it('1D × 1D → scalar (dot product)', async () => {
      const t = matmul(tensor([1, 2, 3]), tensor([4, 5, 6]));
      expect(t.shape).toEqual([]);
      const y = await run(t, { engine });
      expect(await y.allclose(tensor([32], { shape: [] }))).toBe(true);
    });

    it('1D × 2D → shape [N]', async () => {
      const t = matmul(
        tensor([1, 2, 3]),
        tensor([
          [1, 0],
          [0, 1],
          [1, 1],
        ]),
      );
      expect(t.shape).toEqual([2]);
      const y = await run(t, { engine });
      expect(await y.allclose(tensor([4, 5]))).toBe(true);
    });

    it('2D × 1D → shape [M]', async () => {
      const t = matmul(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        tensor([1, 0, 1]),
      );
      expect(t.shape).toEqual([2]);
      const y = await run(t, { engine });
      expect(await y.allclose(tensor([4, 10]))).toBe(true);
    });

    it('3D × 1D → shape [batch, M]', async () => {
      const y = await run(
        matmul(
          tensor([
            [
              [1, 2],
              [3, 4],
            ],
            [
              [5, 6],
              [7, 8],
            ],
          ]),
          tensor([1, 1]),
        ),
        { engine },
      );
      expect(
        await y.allclose(
          tensor([
            [3, 7],
            [11, 15],
          ]),
        ),
      ).toBe(true);
    });

    it('1D × 3D → shape [batch, N]', async () => {
      const y = await run(
        matmul(
          tensor([1, 1]),
          tensor([
            [
              [1, 2],
              [3, 4],
            ],
            [
              [5, 6],
              [7, 8],
            ],
          ]),
        ),
        { engine },
      );
      expect(
        await y.allclose(
          tensor([
            [4, 6],
            [12, 14],
          ]),
        ),
      ).toBe(true);
    });

    it('batched [2,2,3] × [2,3,2] = [2,2,2]', async () => {
      const y = await run(
        matmul(
          tensor([
            [
              [1, 2, 3],
              [4, 5, 6],
            ],
            [
              [7, 8, 9],
              [10, 11, 12],
            ],
          ]),
          tensor([
            [
              [1, 0],
              [0, 1],
              [1, 1],
            ],
            [
              [2, 0],
              [0, 2],
              [1, 1],
            ],
          ]),
        ),
        { engine },
      );
      expect(
        await y.allclose(
          tensor([
            [
              [4, 5],
              [10, 11],
            ],
            [
              [23, 25],
              [32, 34],
            ],
          ]),
        ),
      ).toBe(true);
    });

    it('broadcast batch [1,2,3] × [4,3,1] = [4,2,1]', async () => {
      const a = tensor([
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
      ]); // [1,2,3]
      const b = tensor([
        [[1], [1], [1]],
        [[2], [2], [2]],
        [[0], [0], [0]],
        [[-1], [-1], [-1]],
      ]); // [4,3,1]
      const y = await run(matmul(a, b), { engine });
      expect(
        await y.allclose(
          tensor([
            [[6], [15]],
            [[12], [30]],
            [[0], [0]],
            [[-6], [-15]],
          ]),
        ),
      ).toBe(true);
    });

    it('batched rank-4 [2,1,2,3] × [1,3,3,2] = [2,3,2,2]', async () => {
      const a = tensor([
        [
          [
            [1, 0, 0],
            [0, 1, 0],
          ],
        ],
        [
          [
            [0, 1, 0],
            [0, 0, 1],
          ],
        ],
      ]);
      const b = tensor([
        [
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ],
          [
            [7, 8],
            [9, 10],
            [11, 12],
          ],
          [
            [0, 0],
            [0, 0],
            [0, 0],
          ],
        ],
      ]);
      const t = matmul(a, b);
      expect(t.shape).toEqual([2, 3, 2, 2]);
      const y = await run(t, { engine });
      expect(y.data!.length).toBe(2 * 3 * 2 * 2);
    });

    it('large 32×32 of all ones (each output = 32)', async () => {
      const row = Array.from({ length: 32 }, () => 1.0);
      const A = tensor(Array.from({ length: 32 }, () => row));
      const B = tensor(Array.from({ length: 32 }, () => row));
      const y = await run(matmul(A, B), { engine });
      const expected = tensor(
        Array.from({ length: 1024 }, () => 32.0),
        { shape: [32, 32] },
      );
      expect(await y.allclose(expected, { atol: 1e-3 })).toBe(true);
    });
  });
});

describe('matmul — shape errors', () => {
  it('K-dimension mismatch throws', () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    expect(() => matmul(a, b)).toThrow(/MatMul inner dimensions must match/);
  });
});
