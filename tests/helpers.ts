import { expect } from 'vitest';
import { tensor, compileGraph } from '../packages/core/src';
import type { Tensor } from '../packages/core/src';
import type { NestedArray } from '../packages/core/src';
import { Engine, Backend } from '../packages/runtime/src';
import { CPUBackend } from '../packages/backend-cpu/src';
import { WASMBackend } from '../packages/backend-wasm/src';
import { WebGPUBackend } from '../packages/backend-webgpu/src';

export const BACKENDS = [
  { name: 'CPU',    create: async () => new CPUBackend() as Backend },
  { name: 'WASM',   create: async () => await WASMBackend.create() as Backend },
  { name: 'WebGPU', create: async () => await WebGPUBackend.create() as Backend },
];

async function runGraph(backend: Backend, y: Tensor): Promise<Float32Array> {
  const graph = compileGraph([y]);
  const engine = new Engine(backend);
  engine.evaluate(graph);
  const out = await engine.get(y.id);
  if (!out) throw new Error(`engine.get(${y.id}) returned undefined`);
  return out as Float32Array;
}

export async function runUnary(
  backend: Backend,
  opFn: (a: Tensor) => Tensor,
  data: NestedArray<number>
): Promise<Float32Array> {
  const a = tensor(data);
  return runGraph(backend, opFn(a));
}

export async function runBinary(
  backend: Backend,
  opFn: (a: Tensor, b: Tensor) => Tensor,
  dataA: NestedArray<number>,
  dataB: NestedArray<number>
): Promise<Float32Array> {
  const a = tensor(dataA);
  const b = tensor(dataB);
  return runGraph(backend, opFn(a, b));
}

export function expectClose(actual: Float32Array, expected: number[], tol = 1e-5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i] - expected[i])).toBeLessThan(tol);
  }
}
