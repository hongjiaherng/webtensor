import { Tensor } from '../tensor';
import { InitOptions, buildFromBuffer, totalElements, xorshift32 } from './_internal';

/** Uniform random in `[low, high)` (default `[0, 1)`). Mirrors `torch.rand`. */
export function rand(
  shape: (number | null)[],
  options?: InitOptions & { seed?: number; low?: number; high?: number },
): Tensor {
  const n = totalElements(shape);
  const buffer = new Float32Array(n);
  const next = options?.seed === undefined ? Math.random : xorshift32(options.seed);
  const low = options?.low ?? 0;
  const high = options?.high ?? 1;
  const span = high - low;
  for (let i = 0; i < n; i++) buffer[i] = low + next() * span;
  return buildFromBuffer(shape, buffer, { ...options, dtype: 'float32' });
}
