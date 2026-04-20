import { DType } from '@webtensor/ir';

export type TypedArray = Float32Array | Int32Array | Uint8Array;

export function bytesPerElement(dtype: DType): number {
  switch (dtype) {
    case 'float32':
      return 4;
    case 'int32':
      return 4;
    case 'bool':
      return 1;
  }
}

export function typedArrayCtor(
  dtype: DType,
): Float32ArrayConstructor | Int32ArrayConstructor | Uint8ArrayConstructor {
  switch (dtype) {
    case 'float32':
      return Float32Array;
    case 'int32':
      return Int32Array;
    case 'bool':
      return Uint8Array;
  }
}

export function copyBuffer(dst: TypedArray, src: ArrayBufferView): void {
  // When `src` is a typed array we require it to match `dst`'s element type.
  // Reinterpreting bytes across dtypes (e.g. Float32 → Int32) silently
  // corrupts values and has no legitimate caller in this codebase.
  // A bare `DataView` or `ArrayBuffer` is accepted as an opaque byte source.
  if (ArrayBuffer.isView(src) && !(src instanceof DataView)) {
    const srcCtor = (src as unknown as { constructor: unknown }).constructor;
    const dstCtor = (dst as unknown as { constructor: unknown }).constructor;
    if (srcCtor !== dstCtor) {
      throw new Error(
        `copyBuffer: src type ${(srcCtor as { name: string }).name} does not match dst type ${(dstCtor as { name: string }).name}`,
      );
    }
  }
  if (dst instanceof Float32Array) {
    dst.set(new Float32Array(src.buffer, src.byteOffset, src.byteLength / 4));
  } else if (dst instanceof Int32Array) {
    dst.set(new Int32Array(src.buffer, src.byteOffset, src.byteLength / 4));
  } else {
    dst.set(new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
  }
}

// ---------------------------------------------------------------------------
// Dtype promotion — used at graph-build time by core ops.
// "Wider" of the two dtypes wins: bool < int32 < float32.

const DTYPE_RANK: Record<DType, number> = {
  bool: 0,
  int32: 1,
  float32: 2,
};

/**
 * Return the wider of two dtypes. Examples:
 *   add(float32, int32) → float32
 *   add(bool, int32)    → int32
 *   add(float32, float32) → float32
 */
export function resultDType(a: DType, b: DType): DType {
  return DTYPE_RANK[a] >= DTYPE_RANK[b] ? a : b;
}

/**
 * True iff a dtype supports arithmetic ops (add/sub/mul/div/matmul).
 * `bool` does not — `add(bool, bool)` is rejected; cast first.
 */
export function isArithmeticDType(d: DType): boolean {
  return d === 'float32' || d === 'int32';
}
