import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, mean, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../helpers';

BACKENDS.forEach(({ name, create }) => {
  describe(`mean — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('rank 1: mean all', async () => {
      const y = await run(mean(tensor([1, 2, 3, 4])), { engine });
      expect(y.allclose(tensor([2.5]))).toBe(true);
    });

    it('rank 2: mean all', async () => {
      const y = await run(
        mean(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
        ),
        { engine },
      );
      expect(y.allclose(tensor([3.5]))).toBe(true);
    });

    it('rank 2: axis 0', async () => {
      const y = await run(
        mean(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          0,
        ),
        { engine },
      );
      expect(y.allclose(tensor([2.5, 3.5, 4.5]))).toBe(true);
    });

    it('rank 2: axis 1 keepdim=true', async () => {
      const y = await run(
        mean(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          1,
          true,
        ),
        { engine },
      );
      expect(y.allclose(tensor([[2], [5]]))).toBe(true);
    });

    it('rank 3: axes [0,2]', async () => {
      const y = await run(
        mean(
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
          [0, 2],
        ),
        { engine },
      );
      expect(y.allclose(tensor([3.5, 5.5]))).toBe(true);
    });

    it('rank 3: axis -1', async () => {
      const y = await run(
        mean(
          tensor([
            [
              [1, 2, 3],
              [4, 5, 6],
            ],
          ]),
          -1,
        ),
        { engine },
      );
      expect(y.allclose(tensor([[2, 5]]))).toBe(true);
    });

    it('single element', async () => {
      const y = await run(mean(tensor([[7]])), { engine });
      expect(y.allclose(tensor([7]))).toBe(true);
    });

    it('zeros', async () => {
      const y = await run(
        mean(
          tensor([
            [0, 0, 0],
            [0, 0, 0],
          ]),
        ),
        { engine },
      );
      expect(y.allclose(tensor([0]))).toBe(true);
    });
  });
});

BACKENDS.forEach(({ name, create }) => {
  describe(`mean — autograd — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('gradient of mean over all axes is 1/N everywhere', async () => {
      const a = tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { requiresGrad: true },
      );
      const y = mean(a);
      y.backward();
      const g = await run(a.grad!, { engine });
      expect(
        g.allclose(
          tensor([
            [0.25, 0.25],
            [0.25, 0.25],
          ]),
        ),
      ).toBe(true);
    });
  });
});
