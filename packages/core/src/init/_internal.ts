import { DType } from '@webtensor/ir';
import { TypedArray, typedArrayCtor } from '@webtensor/runtime';
import { Tensor } from '../tensor';
import { Device } from '../types';

export interface InitOptions {
  shape?: (number | null)[];
  dtype?: DType;
  device?: Device;
  /**
   * When true: creates a trainable tensor (a `Placeholder` with `.data` attached
   * to hold the initial values). `compile()` auto-feeds it each call; the
   * optimizer mutates `.data` in place via `opt.step(...)`.
   *
   * When false (default): creates a `Constant` — data is embedded in the graph
   * and cannot be updated by training.
   */
  requiresGrad?: boolean;
}

export function makeTypedArray(dtype: DType, n: number): TypedArray {
  const Ctor = typedArrayCtor(dtype);
  return new Ctor(n);
}

export function totalElements(shape: (number | null)[]): number {
  return shape.reduce<number>((acc, d) => acc * (d ?? 1), 1);
}

/**
 * Build a `Constant` tensor from a concrete typed buffer. Data is embedded
 * directly into the graph node — immutable at runtime.
 */
function constant(
  shape: (number | null)[],
  dtype: DType,
  buffer: TypedArray,
  device?: Device,
): Tensor {
  const t = new Tensor({
    shape,
    dtype,
    device,
    requiresGrad: false,
    ctx: {
      op: 'Constant',
      inputs: [],
      attributes: { data: buffer },
    },
  });
  // Attach .data so users can pass this tensor as a feed to `compile()` —
  // identical buffer is reused, no copy needed.
  t.data = buffer;
  return t;
}

/**
 * Build a trainable `Placeholder` tensor with `.data` carrying initial values.
 * `compile()` auto-feeds `.data` on every call; `opt.step()` mutates it in place.
 */
function parameter(
  shape: (number | null)[],
  dtype: DType,
  buffer: TypedArray,
  device?: Device,
): Tensor {
  const t = new Tensor({
    shape,
    dtype,
    device,
    requiresGrad: true,
    ctx: {
      op: 'Placeholder',
      inputs: [],
      // `data` is embedded as a default value: the engine uses it when no feed
      // is supplied (eager `evaluate(graph)`). compile() feeds the same buffer
      // explicitly each call. Either way, optimizer mutations to `.data` are
      // picked up because attributes.data references the same Float32Array.
      attributes: { shape: shape.map((d) => d ?? 1), data: buffer },
    },
  });
  t.data = buffer;
  return t;
}

/** Dispatch to `constant` or `parameter` based on `requiresGrad`. */
export function buildFromBuffer(
  shape: (number | null)[],
  buffer: TypedArray,
  options?: InitOptions,
): Tensor {
  const dtype = options?.dtype ?? 'float32';
  return options?.requiresGrad
    ? parameter(shape, dtype, buffer, options?.device)
    : constant(shape, dtype, buffer, options?.device);
}

export function xorshift32(seed: number): () => number {
  let s = seed | 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
}
