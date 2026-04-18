import { describe, it, expect, beforeAll } from 'vitest';
import { WebGPUBackend } from '@webtensor/backend-webgpu';

describe('WebGPU bool storage round-trip', () => {
  let backend: WebGPUBackend;
  beforeAll(async () => {
    backend = await WebGPUBackend.create();
  });

  it('write(Uint8Array) + read returns byte-exact Uint8Array', async () => {
    const shape = [2, 3];
    const t = backend.allocate(shape, 'bool');
    const src = new Uint8Array([1, 0, 1, 0, 1, 1]);
    backend.write(t, src);

    const out = (await backend.read(t)) as Uint8Array;
    expect(out).toBeInstanceOf(Uint8Array);
    expect(out.length).toBe(src.length);
    for (let i = 0; i < src.length; i++) expect(out[i]).toBe(src[i]);
  });

  it('device storage is 4 B/elem (bool buffer sized as u32)', () => {
    const t = backend.allocate([5], 'bool');
    expect(t.storage.byteLength).toBe(20);
  });
});
