import { DType } from '@webtensor/ir';

export type TypedArray = Float32Array | Int32Array | Uint8Array;

export function bytesPerElement(dtype: DType): number {
  switch (dtype) {
    case 'float32': return 4;
    case 'int32': return 4;
    case 'bool': return 1;
  }
}

export function typedArrayCtor(dtype: DType): Float32ArrayConstructor | Int32ArrayConstructor | Uint8ArrayConstructor {
  switch (dtype) {
    case 'float32': return Float32Array;
    case 'int32': return Int32Array;
    case 'bool': return Uint8Array;
  }
}

export function copyBuffer(dst: TypedArray, src: ArrayBufferView): void {
  if (dst instanceof Float32Array) {
    dst.set(new Float32Array(src.buffer, src.byteOffset, src.byteLength / 4));
  } else if (dst instanceof Int32Array) {
    dst.set(new Int32Array(src.buffer, src.byteOffset, src.byteLength / 4));
  } else {
    dst.set(new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
  }
}
