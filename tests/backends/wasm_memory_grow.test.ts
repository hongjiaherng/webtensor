import { describe, it, expect, beforeAll } from 'vitest';
import { ones, add, run } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { WASMBackend } from '@webtensor/backend-wasm';

// WASM linear memory grows via `memory.grow` when `__wbindgen_malloc` needs
// more pages. Every grow detaches the underlying ArrayBuffer — any JS
// TypedArray view created from `module.memory.buffer` before the grow becomes
// zero-length. The backend must re-grab a fresh view after each kernel call
// rather than caching one. This test runs a sequence of increasingly large
// allocations through the backend to force multiple grows and verifies that
// results remain correct.

describe('WASM memory grow stress', () => {
  let engine: Engine;
  let initialBufferByteLength: number;

  beforeAll(async () => {
    const backend = await WASMBackend.create();
    engine = new Engine(backend);
    initialBufferByteLength = (backend as unknown as { module: { memory: WebAssembly.Memory } })
      .module.memory.buffer.byteLength;
  });

  it('stays correct across repeated large allocations that trigger memory.grow', async () => {
    const sizes = [1 << 16, 1 << 18, 1 << 20, 1 << 22]; // up to 4M floats = 16 MB
    for (const size of sizes) {
      const a = ones([size]);
      const b = ones([size]);
      const y = await run(add(a, b), { engine });
      const data = y.data as Float32Array;
      expect(data.length).toBe(size);
      expect(data[0]).toBe(2);
      expect(data[size - 1]).toBe(2);
      expect(data[size >> 1]).toBe(2);
    }

    const finalBufferByteLength = (
      engine as unknown as { backend: { module: { memory: WebAssembly.Memory } } }
    ).backend.module.memory.buffer.byteLength;
    // Ensure the stress actually forced growth — otherwise the test is a no-op.
    expect(finalBufferByteLength).toBeGreaterThan(initialBufferByteLength);
  });
});
