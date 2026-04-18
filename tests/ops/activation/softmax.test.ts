import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, sum, run } from '@webtensor/core';
import { softmax } from '@webtensor/nn';
import { Engine } from '@webtensor/runtime';
import { BACKENDS } from '../../helpers';

function jsRefSoftmax(data: number[], shape: number[], axis: number): number[] {
  if (axis < 0) axis = shape.length + axis;
  const total = shape.reduce((a, b) => a * b, 1);
  const axisLen = shape[axis];
  const sliceCount = total / axisLen;
  const out = new Array<number>(total);
  const strides: number[] = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
  const coord: number[] = new Array(shape.length).fill(0);
  for (let s = 0; s < sliceCount; s++) {
    let rem = s;
    for (let d = shape.length - 1; d >= 0; d--) {
      if (d === axis) continue;
      coord[d] = rem % shape[d];
      rem = Math.floor(rem / shape[d]);
    }
    coord[axis] = 0;
    let base = 0;
    for (let d = 0; d < shape.length; d++) base += coord[d] * strides[d];
    let maxV = -Infinity;
    for (let k = 0; k < axisLen; k++) {
      const v = data[base + k * strides[axis]];
      if (v > maxV) maxV = v;
    }
    let sumV = 0;
    const es = new Array<number>(axisLen);
    for (let k = 0; k < axisLen; k++) {
      es[k] = Math.exp(data[base + k * strides[axis]] - maxV);
      sumV += es[k];
    }
    for (let k = 0; k < axisLen; k++) {
      out[base + k * strides[axis]] = es[k] / sumV;
    }
  }
  return out;
}

BACKENDS.forEach(({ name, create }) => {
  describe(`softmax — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('rank 1 (default axis = -1)', async () => {
      const input = [1, 2, 3];
      const y = await run(softmax(tensor(input)), { engine });
      expect(y.allclose(tensor(jsRefSoftmax(input, [3], -1)))).toBe(true);
    });

    it('rank 2 axis = -1', async () => {
      const input = [1, 2, 3, 4, 5, 6];
      const y = await run(
        softmax(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          -1,
        ),
        { engine },
      );
      expect(y.allclose(tensor(jsRefSoftmax(input, [2, 3], -1), { shape: [2, 3] }))).toBe(true);
    });

    it('rank 2 axis = 0', async () => {
      const input = [1, 2, 3, 4, 5, 6];
      const y = await run(
        softmax(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          0,
        ),
        { engine },
      );
      expect(y.allclose(tensor(jsRefSoftmax(input, [2, 3], 0), { shape: [2, 3] }))).toBe(true);
    });

    it('rank 3 middle axis', async () => {
      const flat = Array.from({ length: 2 * 3 * 4 }, (_, i) => (i - 10) * 0.1);
      const nested: number[][][] = [];
      for (let i = 0; i < 2; i++) {
        nested.push([]);
        for (let j = 0; j < 3; j++) {
          nested[i].push([]);
          for (let k = 0; k < 4; k++) nested[i][j].push(flat[i * 12 + j * 4 + k]);
        }
      }
      const y = await run(softmax(tensor(nested), 1), { engine });
      expect(y.allclose(tensor(jsRefSoftmax(flat, [2, 3, 4], 1), { shape: [2, 3, 4] }))).toBe(true);
    });

    it('length-1 axis yields 1.0', async () => {
      const y = await run(softmax(tensor([[[5], [6]]]), -1), { engine });
      expect(y.allclose(tensor([[[1], [1]]]))).toBe(true);
    });

    it('large inputs are numerically stable', async () => {
      const y = await run(softmax(tensor([1000, 1001, 1002])), { engine });
      const arr = Array.from(y.data!);
      arr.forEach((v) => expect(Number.isFinite(v)).toBe(true));
      expect(Math.abs(arr.reduce((a, v) => a + v, 0) - 1)).toBeLessThan(1e-5);
    });

    it('all equal values → uniform', async () => {
      const y = await run(softmax(tensor([[3, 3, 3, 3]])), { engine });
      expect(y.allclose(tensor([[0.25, 0.25, 0.25, 0.25]]))).toBe(true);
    });
  });
});

BACKENDS.forEach(({ name, create }) => {
  describe(`softmax — autograd — ${name}`, () => {
    let engine: Engine;
    beforeAll(async () => {
      engine = new Engine(await create());
    });

    it('grad of softmax(x) · 1 is zero everywhere (saturated)', async () => {
      const a = tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { requiresGrad: true },
      );
      const loss = sum(softmax(a, -1));
      loss.backward();
      const g = await run(a.grad!, { engine });
      expect(
        g.allclose(
          tensor([
            [0, 0, 0],
            [0, 0, 0],
          ]),
          { atol: 1e-5 },
        ),
      ).toBe(true);
    });
  });
});
