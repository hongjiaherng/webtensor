import { expect } from 'vitest';
import { DType } from '@webtensor/ir';
import { Backend, TypedArray } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';
import { WASMBackend } from '@webtensor/backend-wasm';
import { WebGPUBackend } from '@webtensor/backend-webgpu';

/**
 * The backend matrix used by every cross-backend test.
 *
 * Tests iterate `BACKENDS.forEach(...)` and pass `{ engine: new Engine(backend) }`
 * to `run()` / `compile()` so the same user-level code runs on all three backends.
 */
export const BACKENDS = [
  { name: 'CPU', create: async () => (await CPUBackend.create()) as Backend },
  { name: 'WASM', create: async () => (await WASMBackend.create()) as Backend },
  { name: 'WebGPU', create: async () => (await WebGPUBackend.create()) as Backend },
];

/**
 * Dtype axis for tests that need dtype × backend coverage. Tests nest
 * `DTYPES.forEach` inside `BACKENDS.forEach` — no dedicated iterator helper,
 * matching the existing parametric style.
 */
export const DTYPES: readonly DType[] = ['float32', 'int32'] as const;

/** Assert two numeric arrays are equal within a tolerance. */
export function expectClose(actual: TypedArray, expected: number[], tol = 1e-5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i] - expected[i])).toBeLessThan(tol);
  }
}
