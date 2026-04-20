import { describe, it, beforeAll } from 'vitest';
import {
  tensor,
  compileGraph,
  add,
  sub,
  mul,
  div,
  matmul,
  sum,
  mean,
  cast,
  concat,
  pad,
  eq,
  lt,
  isclose,
  relu,
  softmax,
} from '@webtensor/core';
import { Engine, type Backend } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';
import { WASMBackend } from '@webtensor/backend-wasm';
import { WebGPUBackend } from '@webtensor/backend-webgpu';
import { expectClose } from '../helpers';

/**
 * Cross-backend consistency tests.
 * CPU is the oracle. WASM and WebGPU results must match CPU within 1e-5.
 *
 * Each describe block:
 *  - builds a graph once inside beforeAll (pure IR, safe to reuse)
 *  - evaluates on all supported backends
 *  - stores results in a Map<name, Float32Array>
 *  - it blocks compare non-CPU backends against CPU
 */

async function collectResults(
  graphFn: () => ReturnType<typeof tensor>,
  backendNames: string[],
): Promise<Map<string, Float32Array>> {
  const r = await collectAny(graphFn, backendNames);
  const results = new Map<string, Float32Array>();
  r.forEach((v, k) => results.set(k, v as Float32Array));
  return results;
}

async function collectAny(
  graphFn: () => ReturnType<typeof tensor>,
  backendNames: string[],
): Promise<Map<string, ArrayBufferView>> {
  const backends: Record<string, () => Promise<Backend>> = {
    CPU: async () => await CPUBackend.create(),
    WASM: async () => await WASMBackend.create(),
    WebGPU: async () => await WebGPUBackend.create(),
  };

  const results = new Map<string, ArrayBufferView>();

  for (const name of backendNames) {
    const y = graphFn();
    const graph = compileGraph([y]);
    const backend = await backends[name]();
    const engine = new Engine(backend);
    await engine.evaluate(graph);
    const out = await engine.get(y.id);
    results.set(name, out);
  }

  return results;
}

function expectBytesEqual(actual: ArrayBufferView, expected: ArrayBufferView): void {
  const a = new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength);
  const b = new Uint8Array(expected.buffer, expected.byteOffset, expected.byteLength);
  if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) throw new Error(`byte ${i}: ${a[i]} vs ${b[i]}`);
  }
}

// ---------------------------------------------------------------------------

describe('Consistency: Add', () => {
  const results = new Map<string, Float32Array>();

  beforeAll(async () => {
    const r = await collectResults(
      () => add(tensor([1, 2, 3, 4]), tensor([5, 6, 7, 8])),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });

  it('WASM matches CPU', () => expectClose(results.get('WASM')!, [6, 8, 10, 12]));
  it('WebGPU matches CPU', () => expectClose(results.get('WebGPU')!, [6, 8, 10, 12]));
});

// ---------------------------------------------------------------------------

describe('Consistency: Sub', () => {
  const results = new Map<string, Float32Array>();

  beforeAll(async () => {
    const r = await collectResults(
      () => sub(tensor([10, 20, 30]), tensor([1, 2, 3])),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });

  it('WASM matches CPU', () => expectClose(results.get('WASM')!, [9, 18, 27]));
  it('WebGPU matches CPU', () => expectClose(results.get('WebGPU')!, [9, 18, 27]));
});

// ---------------------------------------------------------------------------

describe('Consistency: Mul', () => {
  const results = new Map<string, Float32Array>();

  beforeAll(async () => {
    const r = await collectResults(
      () => mul(tensor([2, 3, 4]), tensor([5, 6, 7])),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });

  it('WASM matches CPU', () => expectClose(results.get('WASM')!, [10, 18, 28]));
  it('WebGPU matches CPU', () => expectClose(results.get('WebGPU')!, [10, 18, 28]));
});

// ---------------------------------------------------------------------------

describe('Consistency: Div', () => {
  const results = new Map<string, Float32Array>();

  beforeAll(async () => {
    const r = await collectResults(
      () => div(tensor([10, 6, 8]), tensor([2, 3, 4])),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });

  it('WASM matches CPU', () => expectClose(results.get('WASM')!, [5, 2, 2]));
  it('WebGPU matches CPU', () => expectClose(results.get('WebGPU')!, [5, 2, 2]));
});

// ---------------------------------------------------------------------------

describe('Consistency: MatMul', () => {
  const results = new Map<string, Float32Array>();

  beforeAll(async () => {
    const r = await collectResults(
      () =>
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
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });

  it('WASM matches CPU', () => expectClose(results.get('WASM')!, [19, 22, 43, 50]));
  it('WebGPU matches CPU', () => expectClose(results.get('WebGPU')!, [19, 22, 43, 50]));
});

// ---------------------------------------------------------------------------

describe('Consistency: Relu', () => {
  const results = new Map<string, Float32Array>();

  beforeAll(async () => {
    const r = await collectResults(
      () => relu(tensor([-1, 0, 1, 2, -3])),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });

  it('WASM matches CPU', () => expectClose(results.get('WASM')!, [0, 0, 1, 2, 0]));
  it('WebGPU matches CPU', () => expectClose(results.get('WebGPU')!, [0, 0, 1, 2, 0]));
});

// ---------------------------------------------------------------------------

describe('Consistency: Transpose', () => {
  const results = new Map<string, Float32Array>();

  beforeAll(async () => {
    const r = await collectResults(
      () =>
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
          .transpose()
          .contiguous(),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });

  it('WASM matches CPU', () => expectClose(results.get('WASM')!, [1, 4, 2, 5, 3, 6]));
  it('WebGPU matches CPU', () => expectClose(results.get('WebGPU')!, [1, 4, 2, 5, 3, 6]));
});

// ---------------------------------------------------------------------------

describe('Consistency: Composite pipeline relu(matmul(x,W) + b)', () => {
  // x:[2,4]  W:[4,3]  b:[3]
  // MatMul → [2,3]: [[0,-2,0],[0,2,0]]
  // + bias [0.5,0.5,0.5] → [[0.5,-1.5,0.5],[0.5,2.5,0.5]]
  // Relu → [[0.5,0,0.5],[0.5,2.5,0.5]]
  const results = new Map<string, Float32Array>();

  beforeAll(async () => {
    const r = await collectResults(() => {
      const x = tensor([
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
      ]);
      const W = tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
      ]);
      const b = tensor([0.5, 0.5, 0.5]);
      return relu(add(matmul(x, W), b));
    }, ['CPU', 'WASM', 'WebGPU']);
    r.forEach((v, k) => results.set(k, v));
  });

  it('CPU produces correct output', () => {
    expectClose(results.get('CPU')!, [0.5, 0, 0.5, 0.5, 2.5, 0.5]);
  });
  it('WASM matches CPU', () => {
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]);
  });
  it('WebGPU matches CPU', () => {
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]);
  });
});

// ---------------------------------------------------------------------------

describe('Consistency: ReduceSum axis 0', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
        sum(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          0,
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: ReduceSum rank-3 axes [0,2]', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
        sum(
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
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: ReduceMean all axes', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
        mean(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: Softmax axis -1 rank 2', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
        softmax(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          -1,
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: Softmax axis 0 rank 3', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
        softmax(
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
          0,
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: Batched MatMul [2,2,3] × [2,3,2]', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
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
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: 1D @ 1D (dot)', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () => matmul(tensor([1, 2, 3]), tensor([4, 5, 6])),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: 1D @ 2D (vector × matrix)', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
        matmul(
          tensor([1, 2, 3]),
          tensor([
            [1, 0, 2],
            [0, 1, 3],
            [1, 1, 4],
          ]),
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: 2D @ 1D (matrix × vector)', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
        matmul(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          tensor([7, 8, 9]),
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

// ---------------------------------------------------------------------------
// Phase 5 parity — ops added across CPU/WASM/WebGPU in phases 1e–5.

describe('Consistency: Concat axis 1', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
        concat(
          [
            tensor([
              [1, 2],
              [3, 4],
            ]),
            tensor([
              [5, 6, 7],
              [8, 9, 10],
            ]),
          ],
          1,
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: Pad constant 3.14', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const r = await collectResults(
      () =>
        pad(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          [1, 2, 2, 1],
          3.14,
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});

describe('Consistency: Cast float32 → bool (truthiness)', () => {
  const results = new Map<string, ArrayBufferView>();
  beforeAll(async () => {
    const r = await collectAny(
      () => cast(tensor([0.0, 1.5, -0.0, 0.001, -2.2]), 'bool'),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () => expectBytesEqual(results.get('WASM')!, results.get('CPU')!));
  it('WebGPU matches CPU', () => expectBytesEqual(results.get('WebGPU')!, results.get('CPU')!));
});

describe('Consistency: Equal (bool output)', () => {
  const results = new Map<string, ArrayBufferView>();
  beforeAll(async () => {
    const r = await collectAny(
      () => eq(tensor([1.0, 2.0, 3.0, 4.0]), tensor([1.0, 0.0, 3.0, 5.0])),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () => expectBytesEqual(results.get('WASM')!, results.get('CPU')!));
  it('WebGPU matches CPU', () => expectBytesEqual(results.get('WebGPU')!, results.get('CPU')!));
});

describe('Consistency: Less broadcast', () => {
  const results = new Map<string, ArrayBufferView>();
  beforeAll(async () => {
    const r = await collectAny(
      () =>
        lt(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          tensor([3]),
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () => expectBytesEqual(results.get('WASM')!, results.get('CPU')!));
  it('WebGPU matches CPU', () => expectBytesEqual(results.get('WebGPU')!, results.get('CPU')!));
});

describe('Consistency: IsClose with rtol/atol', () => {
  const results = new Map<string, ArrayBufferView>();
  beforeAll(async () => {
    const r = await collectAny(
      () =>
        isclose(
          tensor([1.0, 2.0, 3.0, Number.POSITIVE_INFINITY]),
          tensor([1.0 + 1e-7, 2.1, 3.0, Number.POSITIVE_INFINITY]),
          { rtol: 1e-5, atol: 1e-8 },
        ),
      ['CPU', 'WASM', 'WebGPU'],
    );
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () => expectBytesEqual(results.get('WASM')!, results.get('CPU')!));
  it('WebGPU matches CPU', () => expectBytesEqual(results.get('WebGPU')!, results.get('CPU')!));
});

describe('Consistency: rank-12 Add + ReduceSum (widened meta)', () => {
  const results = new Map<string, Float32Array>();
  beforeAll(async () => {
    const nested = (values: number[]): unknown => {
      const wrap = (v: number[]): unknown => [[[[[[[[[[[v]]]]]]]]]]];
      return [wrap(values.slice(0, 3))[0], wrap(values.slice(3, 6))[0]];
    };
    const r = await collectResults(() => {
      const a = tensor(nested([1, -2, 3, -4, 5, -6]) as never);
      const b = tensor(nested([10, 20, 30, 40, 50, 60]) as never);
      return sum(add(a, b), 11);
    }, ['CPU', 'WASM', 'WebGPU']);
    r.forEach((v, k) => results.set(k, v));
  });
  it('WASM matches CPU', () =>
    expectClose(results.get('WASM')!, results.get('CPU')! as unknown as number[]));
  it('WebGPU matches CPU', () =>
    expectClose(results.get('WebGPU')!, results.get('CPU')! as unknown as number[]));
});
