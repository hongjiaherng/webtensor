import { describe, it, expect } from 'vitest';
import { Engine } from '@webtensor/runtime';
import { typedArrayCtor, bytesPerElement } from '@webtensor/runtime';

// Import backends to trigger registerBackend() side effects
import '@webtensor/backend-cpu';
import '@webtensor/backend-wasm';
import '@webtensor/backend-webgpu';

// ---------------------------------------------------------------------------
// DType utilities
// ---------------------------------------------------------------------------

describe('DType utilities', () => {
  it('bytesPerElement returns correct sizes', () => {
    expect(bytesPerElement('float32')).toBe(4);
    expect(bytesPerElement('int32')).toBe(4);
    expect(bytesPerElement('bool')).toBe(1);
  });

  it('typedArrayCtor returns correct constructors', () => {
    expect(typedArrayCtor('float32')).toBe(Float32Array);
    expect(typedArrayCtor('int32')).toBe(Int32Array);
    expect(typedArrayCtor('bool')).toBe(Uint8Array);
  });
});

// ---------------------------------------------------------------------------
// Float32 round-trip (allocate → write → read) across backends
// ---------------------------------------------------------------------------

describe('DType: float32 round-trip', () => {
  for (const device of ['cpu', 'wasm', 'webgpu'] as const) {
    it(`${device}: allocate → write → read float32`, async () => {
      const engine = await Engine.create(device);
      const data = new Float32Array([1.5, -2.5, 3.14, 0, -0.001]);
      engine.set('x', data, [5], 'float32');
      const out = await engine.get('x');
      expect(out).toBeDefined();
      const arr = out as Float32Array;
      expect(arr.length).toBe(5);
      for (let i = 0; i < data.length; i++) {
        expect(Math.abs(arr[i] - data[i])).toBeLessThan(1e-5);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// Int32 round-trip on CPU (the simplest backend to verify dtype infra)
// ---------------------------------------------------------------------------

describe('DType: int32 round-trip', () => {
  it('cpu: allocate → write → read int32', async () => {
    const engine = await Engine.create('cpu');
    const data = new Int32Array([1, -2, 2147483647, -2147483648, 0]);
    engine.set('x', data, [5], 'int32');
    const out = await engine.get('x');
    expect(out).toBeDefined();
    const arr = out as Int32Array;
    expect(arr.length).toBe(5);
    for (let i = 0; i < data.length; i++) {
      expect(arr[i]).toBe(data[i]);
    }
  });
});

// ---------------------------------------------------------------------------
// Bool round-trip on CPU
// ---------------------------------------------------------------------------

describe('DType: bool round-trip', () => {
  it('cpu: allocate → write → read bool', async () => {
    const engine = await Engine.create('cpu');
    const data = new Uint8Array([1, 0, 1, 1, 0]);
    engine.set('x', data, [5], 'bool');
    const out = await engine.get('x');
    expect(out).toBeDefined();
    const arr = out as Uint8Array;
    expect(arr.length).toBe(5);
    for (let i = 0; i < data.length; i++) {
      expect(arr[i]).toBe(data[i]);
    }
  });
});

// ---------------------------------------------------------------------------
// 2D int32 round-trip
// ---------------------------------------------------------------------------

describe('DType: int32 2D round-trip', () => {
  it('cpu: allocate → write → read int32 [2,3]', async () => {
    const engine = await Engine.create('cpu');
    const data = new Int32Array([1, 2, 3, 4, 5, 6]);
    engine.set('x', data, [2, 3], 'int32');
    const out = await engine.get('x');
    expect(out).toBeDefined();
    const arr = out as Int32Array;
    expect(arr.length).toBe(6);
    expect(Array.from(arr)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});
