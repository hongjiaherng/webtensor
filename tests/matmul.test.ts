import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, compileGraph, matmul } from '../packages/core/src';
import { Engine, Backend } from '../packages/runtime/src';
import { CPUBackend } from '../packages/backend-cpu/src';
import { WebGPUBackend } from '../packages/backend-webgpu/src';
import { WASMBackend } from '../packages/backend-wasm/src';

const backends = [
  { name: 'CPU', create: async () => new CPUBackend() },
  { name: 'WebGPU', create: async () => await WebGPUBackend.create() },
  { name: 'WASM', create: async () => await WASMBackend.create() }
];

describe('MatMul End-to-End Workflow', () => {
  backends.forEach(({ name, create }) => {
    describe(`${name} Backend`, () => {
      let backend: Backend;

      beforeAll(async () => {
        backend = await create();
      });

      it('should compile and evaluate y = matmul(A, B) natively', async () => {
        // 2x3 Matrix natively extracted
        const a = tensor([
          [1, 2, 3],
          [4, 5, 6]
        ], { requiresGrad: true });

        // 3x2 Matrix natively extracted
        const b = tensor([
          [7, 8],
          [9, 10],
          [11, 12]
        ]);

        const y = matmul(a, b);
        const graph = compileGraph([y]);

        const engine = new Engine(backend);
        engine.evaluate(graph);
        
        const out = await engine.get(y.id) as Float32Array;
        expect(Array.from(out)).toEqual([58, 64, 139, 154]);
        
        // Inject graphical validation into Chromium's test watch pane!
        if (typeof document !== 'undefined') {
          const el = document.createElement('div');
          el.style.padding = '20px';
          el.style.fontFamily = 'monospace';
          el.innerHTML = `
            <h2 style="color: #60a5fa">MatMul Architecture Executed Successfully!</h2>
            <p><strong>Graph:</strong> y = matmul(A, B)</p>
            <p><strong>Shapes:</strong> [2,3] x [3,2] = [2,2]</p>
            <p><strong>${name} Target:</strong> [${Array.from(out).join(', ')}]</p>
          `;
          document.body.appendChild(el);
        }
      });
    });
  });
});
