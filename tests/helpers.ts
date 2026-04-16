import { expect } from 'vitest';
import { tensor, compileGraph } from '@webtensor/core';
import type { Tensor } from '@webtensor/core';
import type { NestedArray } from '@webtensor/core';
import { Engine, Backend, TypedArray } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';
import { WASMBackend } from '@webtensor/backend-wasm';
import { WebGPUBackend } from '@webtensor/backend-webgpu';

export const BACKENDS = [
  { name: 'CPU', create: async () => new CPUBackend() as Backend },
  { name: 'WASM', create: async () => (await WASMBackend.create()) as Backend },
  { name: 'WebGPU', create: async () => (await WebGPUBackend.create()) as Backend },
];

export async function runGraph(backend: Backend, y: Tensor): Promise<TypedArray> {
  const graph = compileGraph([y]);
  const engine = new Engine(backend);
  await engine.evaluate(graph);
  const out = await engine.get(y.id);
  if (!out) throw new Error(`engine.get(${y.id}) returned undefined`);
  return out as TypedArray;
}

export async function runUnary(
  backend: Backend,
  opFn: (a: Tensor) => Tensor,
  data: NestedArray<number>,
): Promise<TypedArray> {
  const a = tensor(data);
  return runGraph(backend, opFn(a));
}

export async function runBinary(
  backend: Backend,
  opFn: (a: Tensor, b: Tensor) => Tensor,
  dataA: NestedArray<number>,
  dataB: NestedArray<number>,
): Promise<TypedArray> {
  const a = tensor(dataA);
  const b = tensor(dataB);
  return runGraph(backend, opFn(a, b));
}

export function expectClose(actual: TypedArray, expected: number[], tol = 1e-5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i] - expected[i])).toBeLessThan(tol);
  }
}
