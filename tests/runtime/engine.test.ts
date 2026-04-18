import { describe, it, expect } from 'vitest';
import { tensor, compileGraph, add } from '@webtensor/core';
import { Engine } from '@webtensor/runtime';
import { expectClose } from '../helpers';

// Import backend packages to trigger registerBackend() side effects
import '@webtensor/backend-cpu';
import '@webtensor/backend-wasm';
import '@webtensor/backend-webgpu';

// ---------------------------------------------------------------------------
// Device dispatch via Engine.create()
// ---------------------------------------------------------------------------

describe('Engine.create() — device dispatch', () => {
  it('creates CPU engine via Engine.create("cpu")', async () => {
    const engine = await Engine.create('cpu');
    expect(engine.device).toBe('cpu');

    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const y = add(a, b);
    const graph = compileGraph([y]);
    await engine.evaluate(graph);
    const out = await engine.get(y.id);
    expectClose(out as Float32Array, [5, 7, 9]);
  });

  it('creates WASM engine via Engine.create("wasm")', async () => {
    const engine = await Engine.create('wasm');
    expect(engine.device).toBe('wasm');

    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const y = add(a, b);
    const graph = compileGraph([y]);
    await engine.evaluate(graph);
    const out = await engine.get(y.id);
    expectClose(out as Float32Array, [5, 7, 9]);
  });

  it('creates WebGPU engine via Engine.create("webgpu")', async () => {
    const engine = await Engine.create('webgpu');
    expect(engine.device).toBe('webgpu');

    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const y = add(a, b);
    const graph = compileGraph([y]);
    await engine.evaluate(graph);
    const out = await engine.get(y.id);
    expectClose(out as Float32Array, [5, 7, 9]);
  });

  it('throws for unknown device', async () => {
    await expect(Engine.create('tpu')).rejects.toThrow(/No backend registered for device 'tpu'/);
  });

  it('all three devices produce identical results', async () => {
    const a = tensor([1, -2, 3, -4]);
    const b = tensor([10, 20, 30, 40]);
    const y = add(a, b);

    const results: Float32Array[] = [];
    for (const device of ['cpu', 'wasm', 'webgpu'] as const) {
      const engine = await Engine.create(device);
      const graph = compileGraph([y]);
      await engine.evaluate(graph);
      results.push((await engine.get(y.id)) as Float32Array);
    }

    expectClose(results[0], [11, 18, 33, 36]);
    expectClose(results[1], Array.from(results[0]));
    expectClose(results[2], Array.from(results[0]));
  });
});

// ---------------------------------------------------------------------------
// Engine memory management
// ---------------------------------------------------------------------------

describe('Engine — memory lifecycle', () => {
  it('set/get round-trip', async () => {
    const engine = await Engine.create('cpu');
    engine.set('x', new Float32Array([1, 2, 3]), [3]);
    const out = await engine.get('x');
    expect(Array.from(out as Float32Array)).toEqual([1, 2, 3]);
  });

  it('get returns undefined for unknown name', async () => {
    const engine = await Engine.create('cpu');
    const out = await engine.get('nonexistent');
    expect(out).toBeUndefined();
  });

  it('dispose removes tensor', async () => {
    const engine = await Engine.create('cpu');
    engine.set('x', new Float32Array([1, 2, 3]), [3]);
    engine.dispose('x');
    const out = await engine.get('x');
    expect(out).toBeUndefined();
  });

  it('set overwrites previous tensor', async () => {
    const engine = await Engine.create('cpu');
    engine.set('x', new Float32Array([1, 2, 3]), [3]);
    engine.set('x', new Float32Array([4, 5, 6]), [3]);
    const out = await engine.get('x');
    expect(Array.from(out as Float32Array)).toEqual([4, 5, 6]);
  });
});
