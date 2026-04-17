import { add, compileGraph, matmul, mul, tensor } from '@webtensor/core';
import { Engine, type Backend } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';
import { WASMBackend } from '@webtensor/backend-wasm';
import { WebGPUBackend } from '@webtensor/backend-webgpu';

export type TestResult = {
  name: string;
  passed: boolean;
  got?: number[];
  expected?: number[];
  error?: string;
};

const TOL = 1e-5;

async function runCase(
  name: string,
  fn: () => Promise<{ got: number[]; expected: number[] }>,
): Promise<TestResult> {
  try {
    const { got, expected } = await fn();
    const passed =
      got.length === expected.length && expected.every((v, i) => Math.abs(got[i]! - v) < TOL);
    return { name, passed, got, expected };
  } catch (e) {
    return { name, passed: false, error: String(e) };
  }
}

async function evalAndRead(engine: Engine, output: { id: string }): Promise<number[]> {
  await engine.evaluate(compileGraph([output as never]));
  const out = (await engine.get(output.id)) as Float32Array;
  return Array.from(out);
}

const cases = (engine: Engine): Promise<TestResult>[] => [
  runCase('add [2×2]', async () => ({
    got: await evalAndRead(
      engine,
      add(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([
          [5, 6],
          [7, 8],
        ]),
      ),
    ),
    expected: [6, 8, 10, 12],
  })),
  runCase('mul [1×3]', async () => ({
    got: await evalAndRead(engine, mul(tensor([2, 3, 4]), tensor([5, 6, 7]))),
    expected: [10, 18, 28],
  })),
  runCase('matmul identity [2×2]', async () => ({
    got: await evalAndRead(
      engine,
      matmul(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([
          [1, 0],
          [0, 1],
        ]),
      ),
    ),
    expected: [1, 2, 3, 4],
  })),
  runCase('broadcast add [2×2] + scalar', async () => ({
    got: await evalAndRead(
      engine,
      add(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([[10]]),
      ),
    ),
    expected: [11, 12, 13, 14],
  })),
];

async function runWith(backend: Backend): Promise<TestResult[]> {
  return Promise.all(cases(new Engine(backend)));
}

export async function runCpuTests(): Promise<TestResult[]> {
  return runWith(new CPUBackend());
}

export async function runWasmTests(): Promise<TestResult[]> {
  try {
    return runWith(await WASMBackend.create());
  } catch (e) {
    return [{ name: 'WASM init', passed: false, error: String(e) }];
  }
}

export async function runWebGpuTests(): Promise<TestResult[]> {
  try {
    return runWith(await WebGPUBackend.create());
  } catch (e) {
    return [{ name: 'WebGPU init', passed: false, error: String(e) }];
  }
}

// Backend availability detection — for the env panel at the top of the page
export type BackendAvailability = {
  name: string;
  available: boolean;
  detail?: string;
};

export async function detectBackends(): Promise<BackendAvailability[]> {
  const cpu: BackendAvailability = { name: 'CPU', available: true, detail: 'always available' };

  const wasm: BackendAvailability = (() => {
    if (typeof WebAssembly === 'undefined') {
      return { name: 'WASM', available: false, detail: 'WebAssembly not supported' };
    }
    return { name: 'WASM', available: true, detail: 'WebAssembly supported' };
  })();

  const webgpu = await (async (): Promise<BackendAvailability> => {
    if (!('gpu' in navigator)) {
      return {
        name: 'WebGPU',
        available: false,
        detail: 'navigator.gpu missing (browser unsupported or insecure context)',
      };
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter)
        return { name: 'WebGPU', available: false, detail: 'no adapter (GPU disabled?)' };
      const info = adapter.info;
      const detail =
        [info.vendor, info.architecture, info.device].filter(Boolean).join(' · ') ||
        'adapter found';
      return { name: 'WebGPU', available: true, detail };
    } catch (e) {
      return { name: 'WebGPU', available: false, detail: String(e) };
    }
  })();

  return [cpu, wasm, webgpu];
}
