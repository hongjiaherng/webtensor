import { Tensor } from './tensor';

/**
 * Shape comparison helper — two tensors have equal shape iff they have the
 * same rank and each dimension matches exactly. `null` dims compare as
 * themselves.
 */
function shapesEqual(a: Tensor, b: Tensor): boolean {
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return false;
  }
  return true;
}

function materialized(t: Tensor, label: string): ArrayLike<number> {
  if (!t.data) {
    throw new Error(
      `${label}: tensor has no .data — only evaluated tensors can be compared. ` +
        'Call `await run(t)` or `compile(...)` first, or build the tensor with a factory like `tensor()`.',
    );
  }
  return t.data as ArrayLike<number>;
}

export interface AllcloseOptions {
  /** Relative tolerance. Default: 1e-5 (PyTorch / JAX default). */
  rtol?: number;
  /** Absolute tolerance. Default: 1e-8. */
  atol?: number;
  /** If true, NaN values in the same positions compare equal. Default: false. */
  equalNan?: boolean;
}

/**
 * Strict element-wise equality. Returns `true` iff the two tensors have the
 * same shape AND every element is exactly equal. Mirrors `torch.equal(a, b)`
 * and `jnp.array_equal(a, b)`.
 *
 * NaN values never compare equal (even to themselves) — matching IEEE 754
 * and the PyTorch/JAX convention.
 */
export function equal(a: Tensor, b: Tensor): boolean {
  if (!shapesEqual(a, b)) return false;
  const da = materialized(a, 'equal');
  const db = materialized(b, 'equal');
  if (da.length !== db.length) return false;
  for (let i = 0; i < da.length; i++) {
    if (da[i] !== db[i]) return false;
  }
  return true;
}

/**
 * Numeric closeness check. Returns `true` iff the two tensors have the same
 * shape AND every pair of elements satisfies
 *   `|a - b| <= atol + rtol * |b|`
 * (matching the NumPy / PyTorch / JAX formula).
 *
 * Defaults: `rtol = 1e-5`, `atol = 1e-8`, `equalNan = false`.
 *
 * Infinity handling: `+∞ ≈ +∞` and `-∞ ≈ -∞`, but `+∞ ≉ -∞` and `±∞ ≉ finite`.
 */
export function allclose(a: Tensor, b: Tensor, opts: AllcloseOptions = {}): boolean {
  if (!shapesEqual(a, b)) return false;
  const rtol = opts.rtol ?? 1e-5;
  const atol = opts.atol ?? 1e-8;
  const equalNan = opts.equalNan ?? false;
  const da = materialized(a, 'allclose');
  const db = materialized(b, 'allclose');
  if (da.length !== db.length) return false;

  for (let i = 0; i < da.length; i++) {
    const x = da[i];
    const y = db[i];

    // NaN handling
    const xNan = Number.isNaN(x);
    const yNan = Number.isNaN(y);
    if (xNan || yNan) {
      if (xNan && yNan && equalNan) continue;
      return false;
    }

    // Infinity: only close if identical (sign matters)
    const xInf = !Number.isFinite(x);
    const yInf = !Number.isFinite(y);
    if (xInf || yInf) {
      if (x !== y) return false;
      continue;
    }

    if (Math.abs(x - y) > atol + rtol * Math.abs(y)) return false;
  }
  return true;
}
