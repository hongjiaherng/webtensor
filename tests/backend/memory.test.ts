import { describe, it, expect, vi, beforeAll, afterEach } from 'vitest';
import { tensor, compileGraph, mul } from '@webtensor/core';
import { Engine, Backend } from '@webtensor/runtime';
import { CPUBackend } from '@webtensor/backend-cpu';
import { WebGPUBackend } from '@webtensor/backend-webgpu';
import { WASMBackend } from '@webtensor/backend-wasm';

const backends = [
  { name: 'CPU', create: async () => new CPUBackend() },
  { name: 'WebGPU', create: async () => await WebGPUBackend.create() },
  { name: 'WASM', create: async () => await WASMBackend.create() },
];

describe('JIT Garbage Collection & Memory Sweeps', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  backends.forEach(({ name, create }) => {
    describe(`${name} Backend`, () => {
      let backend: Backend;

      beforeAll(async () => {
        backend = await create();
      });

      it('should automatically destroy intermediate buffers during evaluation', async () => {
        // 1. Author a multi-layered graph
        const a = tensor([2.0]);
        const b = tensor([3.0]);
        const c = tensor([4.0]);

        // t1 = a + b (Intermediate payload!)
        const t1 = a.add(b);
        // t2 = t1 * c (Final output target!)
        const t2 = mul(t1, c);

        const graph = compileGraph([t2]);

        // Spies let us trace exactly what is destroyed!
        const disposeSpy = vi.spyOn(backend, 'dispose');

        const engine = new Engine(backend);
        await engine.evaluate(graph);

        // Assert the mathematical execution correctly resolved despite cleanup
        const out = (await engine.get(t2.id)) as Float32Array;
        expect(out[0]).toBe(20.0); // (2 + 3) * 4 = 20

        // 2. Validate Automated Memory Sweeping
        // The engine should have exactly deleted ONE tensor implicitly gracefully: the intermediate `t1` node buffer!
        expect(disposeSpy).toHaveBeenCalledTimes(1);

        // 3. Test Manual Disposals for End Output Arrays
        engine.dispose(t2.id);
        expect(disposeSpy).toHaveBeenCalledTimes(2);

        // If we request t2 again, the engine gracefully denies reading physically dead buffers!
        const deadBuffer = await engine.get(t2.id);
        expect(deadBuffer).toBeUndefined();
      });
    });
  });
});
