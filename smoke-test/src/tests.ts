import {
  add,
  compile,
  eq,
  grad,
  matmul,
  mul,
  randn,
  relu,
  run,
  softmax,
  sum,
  tensor,
  type Device,
} from '@webtensor/core';
import { mseLoss, SGD } from '@webtensor/nn';

// Side-effect registers each device with the runtime registry so
// `Engine.create(device)` / `run(t, { device })` work without any
// explicit backend constructor. This is the idiomatic shape.
import '@webtensor/backend-cpu';
import '@webtensor/backend-wasm';
import '@webtensor/backend-webgpu';

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

const toArray = (t: { data?: ArrayBufferView }): number[] =>
  Array.from(t.data as Float32Array | Int32Array);

const cases = (device: Device): Promise<TestResult>[] => [
  runCase('add [2×2]', async () => ({
    got: toArray(
      await run(
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
        { device },
      ),
    ),
    expected: [6, 8, 10, 12],
  })),

  runCase('mul [1×3]', async () => ({
    got: toArray(await run(mul(tensor([2, 3, 4]), tensor([5, 6, 7])), { device })),
    expected: [10, 18, 28],
  })),

  runCase('matmul identity [2×2]', async () => ({
    got: toArray(
      await run(
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
        { device },
      ),
    ),
    expected: [1, 2, 3, 4],
  })),

  runCase('broadcast add [2×2] + scalar', async () => ({
    got: toArray(
      await run(
        add(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          tensor([[10]]),
        ),
        { device },
      ),
    ),
    expected: [11, 12, 13, 14],
  })),

  runCase('sum axis=1', async () => ({
    got: toArray(
      await run(
        sum(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          1,
        ),
        { device },
      ),
    ),
    expected: [6, 15],
  })),

  runCase('softmax [3] sums to 1', async () => {
    const out = await run(softmax(tensor([1, 2, 3])), { device });
    const data = out.data as Float32Array;
    const total = data[0]! + data[1]! + data[2]!;
    return {
      got: [Math.round(total * 1e5) / 1e5],
      expected: [1.0],
    };
  }),

  runCase('relu [4]', async () => ({
    got: toArray(await run(relu(tensor([-1, 0, 1, 2])), { device })),
    expected: [0, 0, 1, 2],
  })),

  runCase('eq (bool) [3]', async () => {
    const out = await run(eq(tensor([1, 2, 3]), tensor([1, 0, 3])), { device });
    const bytes = out.data as Uint8Array;
    return {
      got: Array.from(bytes, (b) => (b ? 1 : 0)),
      expected: [1, 0, 1],
    };
  }),

  runCase('autograd: grad(mseLoss) via compile() + SGD step', async () => {
    const W = randn([2], { requiresGrad: true, std: 0.1 });
    const opt = new SGD(0.5);

    const step = await compile(
      ({ x, y }) => {
        const pred = sum(mul(x, W), -1);
        const loss = mseLoss(pred, y);
        return { loss, dW: grad(loss, W) };
      },
      { x: [4, 2], y: [4] },
      { device },
    );

    // Well-conditioned design matrix: X^T X = 3·I so lr=0.5 converges fast.
    // True weights W* = [2, 3]; y = X @ W*.
    const xs = new Float32Array([1, 0, 0, 1, 1, 1, -1, 1]); // 4×2
    const ys = new Float32Array([2, 3, 5, 1]);

    let finalLoss = Infinity;
    for (let i = 0; i < 60; i++) {
      const { loss, dW } = await step({ x: xs, y: ys });
      opt.step([W], [dW]);
      finalLoss = (loss.data as Float32Array)[0]!;
    }
    return {
      got: [finalLoss < 1e-3 ? 1 : 0],
      expected: [1],
    };
  }),
];

async function runWith(device: Device): Promise<TestResult[]> {
  try {
    return await Promise.all(cases(device));
  } catch (e) {
    return [{ name: `${device} init`, passed: false, error: String(e) }];
  }
}

export async function runCpuTests(): Promise<TestResult[]> {
  return runWith('cpu');
}

export async function runWasmTests(): Promise<TestResult[]> {
  return runWith('wasm');
}

export async function runWebGpuTests(): Promise<TestResult[]> {
  return runWith('webgpu');
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
