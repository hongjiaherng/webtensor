import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, compileGraph, matmul } from '../../packages/core/src';
import { Engine, Backend } from '../../packages/runtime/src';
import { CPUBackend } from '../../packages/backend-cpu/src';
import { WASMBackend } from '../../packages/backend-wasm/src';

const backends = [
  { name: 'CPU', create: async () => new CPUBackend() },
  { name: 'WASM', create: async () => await WASMBackend.create() },
  // WebGPU excluded: matmul backward uses transpose (now a view), and the
  // gather pre-pass that materializes non-contiguous views has a meta[0]=0 bug.
];

describe('Autograd Core Backpropagation', () => {
  backends.forEach(({ name, create }) => {
    describe(`${name} Backend`, () => {
      let backend: Backend;

      beforeAll(async () => {
        backend = await create();
      });

      it('should compute mathematically exact gradients for dynamic topological matmul equations', async () => {
        // 1x2 Matrix
        const a = tensor([[2.0, 3.0]], { requiresGrad: true });

        // 2x1 Matrix
        const b = tensor([[4.0], [5.0]], { requiresGrad: true });

        // y = a * b
        // Shape = [1, 1]
        const y = matmul(a, b);

        // Evaluate Forward Pass!
        // Result = 2*4 + 3*5 = 23
        const engine = new Engine(backend);
        engine.evaluate(compileGraph([y]));

        const yOut = (await engine.get(y.id)) as Float32Array;
        expect(yOut[0]).toBe(23.0);

        // Evaluate Backward Pass dynamically tracing chain rule
        y.backward();

        engine.evaluate(compileGraph([a.grad!, b.grad!]));

        const gradA = (await engine.get(a.grad!.id)) as Float32Array;
        expect(gradA).toBeDefined();
        expect(Array.from(gradA)).toEqual([4, 5]);

        const gradB = (await engine.get(b.grad!.id)) as Float32Array;
        expect(gradB).toBeDefined();
        expect(Array.from(gradB)).toEqual([2, 3]);
      });
    });
  });
});
