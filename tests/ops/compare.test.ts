import { describe, it, expect } from 'vitest';
import { tensor, equal, allclose, add, run, Engine } from '@webtensor/core';
import { CPUBackend } from '@webtensor/backend-cpu';

const engine = new Engine(await CPUBackend.create());

describe('equal', () => {
  it('equal shapes + equal values', () => {
    expect(equal(tensor([1, 2, 3]), tensor([1, 2, 3]))).toBe(true);
  });

  it('equal shapes + different values', () => {
    expect(equal(tensor([1, 2, 3]), tensor([1, 2, 4]))).toBe(false);
  });

  it('different shape → false', () => {
    expect(equal(tensor([[1, 2, 3]]), tensor([1, 2, 3]))).toBe(false);
  });

  it('rank 2', () => {
    expect(
      equal(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([
          [1, 2],
          [3, 4],
        ]),
      ),
    ).toBe(true);
    expect(
      equal(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([
          [1, 2],
          [3, 5],
        ]),
      ),
    ).toBe(false);
  });

  it('NaN never equals NaN', () => {
    expect(equal(tensor([NaN]), tensor([NaN]))).toBe(false);
  });

  it('as a method: .equals()', () => {
    expect(tensor([1, 2, 3]).equals(tensor([1, 2, 3]))).toBe(true);
    expect(tensor([1, 2, 3]).equals(tensor([1, 2]))).toBe(false);
  });

  it('works on evaluated tensors from run()', async () => {
    const y = await run(add(tensor([1, 2, 3]), tensor([10, 20, 30])), { engine });
    expect(y.equals(tensor([11, 22, 33]))).toBe(true);
  });

  it('throws if the tensor has no data', () => {
    // Build an unevaluated tensor: add of two constants (needs run() to materialize)
    const ungraphed = add(tensor([1, 2]), tensor([3, 4]));
    // constants have .data, but the add result doesn't until evaluated
    expect(() => equal(ungraphed, tensor([4, 6]))).toThrow(/no \.data/);
  });
});

describe('allclose', () => {
  it('exactly equal values', () => {
    expect(allclose(tensor([1, 2, 3]), tensor([1, 2, 3]))).toBe(true);
  });

  it('close within default tolerance', () => {
    expect(allclose(tensor([1.0000001]), tensor([1.0]))).toBe(true);
  });

  it('far apart → false', () => {
    expect(allclose(tensor([1, 2, 3]), tensor([1, 2, 4]))).toBe(false);
  });

  it('rtol controls relative tolerance', () => {
    expect(allclose(tensor([1000]), tensor([1001]), { rtol: 1e-2 })).toBe(true);
    expect(allclose(tensor([1000]), tensor([1001]), { rtol: 1e-4 })).toBe(false);
  });

  it('atol controls absolute tolerance', () => {
    expect(allclose(tensor([0]), tensor([0.001]), { atol: 0.01 })).toBe(true);
    expect(allclose(tensor([0]), tensor([0.001]), { atol: 1e-6 })).toBe(false);
  });

  it('different shapes → false', () => {
    expect(allclose(tensor([[1]]), tensor([1]))).toBe(false);
  });

  it('NaN ≠ NaN by default', () => {
    expect(allclose(tensor([NaN]), tensor([NaN]))).toBe(false);
  });

  it('NaN = NaN when equalNan: true', () => {
    expect(allclose(tensor([NaN]), tensor([NaN]), { equalNan: true })).toBe(true);
  });

  it('same-signed infinity compares equal', () => {
    expect(allclose(tensor([Infinity]), tensor([Infinity]))).toBe(true);
    expect(allclose(tensor([-Infinity]), tensor([-Infinity]))).toBe(true);
  });

  it('opposite-signed infinities are not close', () => {
    expect(allclose(tensor([Infinity]), tensor([-Infinity]))).toBe(false);
  });

  it('finite vs infinite is not close', () => {
    expect(allclose(tensor([1e30]), tensor([Infinity]))).toBe(false);
  });

  it('as a method: .allclose()', () => {
    // 1e-7 is well inside the default rtol=1e-5 even after Float32 quantization.
    expect(tensor([1.0, 2.0]).allclose(tensor([1.0000001, 2.0000001]))).toBe(true);
  });

  it('works after run() with small numeric drift', async () => {
    const y = await run(add(tensor([0.1, 0.2]), tensor([0.2, 0.1])), { engine });
    // 0.1 + 0.2 = 0.30000000000000004 in IEEE 754
    expect(y.allclose(tensor([0.3, 0.3]))).toBe(true);
  });

  it('Tensor.run() method evaluates in place of module-level run()', async () => {
    const y = await add(tensor([1, 2, 3]), tensor([10, 20, 30])).run({ engine });
    expect(y.equals(tensor([11, 22, 33]))).toBe(true);
  });
});
