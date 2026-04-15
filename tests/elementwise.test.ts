import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, compileGraph, mul } from '../packages/core/src';
import { Engine, Backend } from '../packages/runtime/src';
import { CPUBackend } from '../packages/backend-cpu/src';
import { WebGPUBackend } from '../packages/backend-webgpu/src';
import { WASMBackend } from '../packages/backend-wasm/src';

const backends = [
  { name: 'CPU', create: async () => new CPUBackend() },
  { name: 'WebGPU', create: async () => await WebGPUBackend.create() },
  { name: 'WASM', create: async () => await WASMBackend.create() }
];

describe('Element-wise Operations Workflow', () => {
  backends.forEach(({ name, create }) => {
    describe(`${name} Backend`, () => {
      let backend: Backend;

      beforeAll(async () => {
        backend = await create();
      });

      it('should compile and evaluate y = (a + b) * c natively', async () => {
        // --- 1. Authoring ---
        const a = tensor([5.0, 5.5], { requiresGrad: true });
        const b = tensor([3.0, 1.5]);
        const c = tensor([10.0, 10.0]);

        const y = mul(a.add(b), c);

        // --- 2. Compilation to IR ---
        const graph = compileGraph([y]);
        
        // Nodes: a, b, c, add, mul = 5 nodes
        expect(graph.nodes.length).toBe(5);

        // --- 3. Evaluate Engine ---
        const engine = new Engine(backend);
        engine.evaluate(graph);
        
        const out = await engine.get(y.id) as Float32Array;
        expect(out).toBeDefined();
        if (out) {
          expect(out.length).toBe(2);
          expect(out[0]).toBe(80.0);
          expect(out[1]).toBe(70.0);
          
          // Inject visual feedback into the browser's DOM for watch mode!
          if (typeof document !== 'undefined') {
            const el = document.createElement('div');
            el.style.padding = '20px';
            el.style.fontFamily = 'monospace';
            el.innerHTML = `
              <h2 style="color: #4ade80">Elementwise Kernels Executed Successfully!</h2>
              <p><strong>Graph:</strong> y = (a + b) * c</p>
              <p><strong>${name} Target:</strong> [${out.join(', ')}]</p>
            `;
            document.body.appendChild(el);
          }
        }
      });
    });
  });
});
