import { Tensor } from '../tensor';
import { InitOptions, buildFromBuffer, totalElements, xorshift32 } from './_internal';

/**
 * Normal (Gaussian) init via Box–Muller. Mirrors `torch.randn`; extends with
 * `mean` and `std` options for Xavier / He-style scaled initialization.
 * @category Factories
 */
export function randn(
  shape: (number | null)[],
  options?: InitOptions & { seed?: number; mean?: number; std?: number },
): Tensor {
  const n = totalElements(shape);
  const buffer = new Float32Array(n);
  const next = options?.seed === undefined ? Math.random : xorshift32(options.seed);
  const mean = options?.mean ?? 0;
  const std = options?.std ?? 1;
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(next(), 1e-12);
    const u2 = next();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    buffer[i] = mean + std * (r * Math.cos(theta));
    if (i + 1 < n) buffer[i + 1] = mean + std * (r * Math.sin(theta));
  }
  return buildFromBuffer(shape, buffer, { ...options, dtype: 'float32' });
}
