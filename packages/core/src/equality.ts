import { Tensor } from './tensor';
import { eq } from './ops/elementwise/eq';
import { isclose } from './ops/elementwise/isclose';
import { cast } from './ops/elementwise/cast';
import { all } from './ops/reduction/all';
import { run, RunOptions } from './run';

/**
 * Shape comparison helper — two tensors have equal shape iff they have the
 * same rank and each dimension matches exactly.
 */
function shapesEqual(a: Tensor, b: Tensor): boolean {
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return false;
  }
  return true;
}

export interface AllcloseOptions {
  /** Relative tolerance. Default: 1e-5. */
  rtol?: number;
  /** Absolute tolerance. Default: 1e-8. */
  atol?: number;
  /** If true, NaN values in the same positions compare equal. Default: false. */
  equalNan?: boolean;
}

/**
 * Strict element-wise equality. Resolves to `true` iff the two tensors have
 * the same shape AND every element is exactly equal.
 *
 * Implemented as `all(eq(a, b))` — comparison runs on the active backend and
 * only the final scalar bool is pulled back to JS.
 *
 * NaN values never compare equal (even to themselves) — matching IEEE 754.
 */
export async function equal(a: Tensor, b: Tensor, opts: RunOptions = {}): Promise<boolean> {
  if (!shapesEqual(a, b)) return false;
  // `eq` requires float32 / int32 inputs; bool is 0/1 so casting to int32
  // preserves semantics without adding bool kernels across three backends.
  const ac = a.dtype === 'bool' ? cast(a, 'int32') : a;
  const bc = b.dtype === 'bool' ? cast(b, 'int32') : b;
  const r = await run(all(eq(ac, bc)), opts);
  return (r.data as Uint8Array)[0] === 1;
}

/**
 * Numeric closeness check. Resolves to `true` iff the two tensors have the
 * same shape AND every pair of elements satisfies
 *   `|a - b| <= atol + rtol * |b|`
 *
 * Defaults: `rtol = 1e-5`, `atol = 1e-8`, `equalNan = false`.
 */
export async function allclose(
  a: Tensor,
  b: Tensor,
  opts: AllcloseOptions & RunOptions = {},
): Promise<boolean> {
  if (!shapesEqual(a, b)) return false;
  const { rtol, atol, equalNan, ...runOpts } = opts;
  const r = await run(all(isclose(a, b, { rtol, atol, equalNan })), runOpts);
  return (r.data as Uint8Array)[0] === 1;
}
