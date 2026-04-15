import { Tensor } from './tensor';
import { DType, Device, NestedArray } from './types';
import { inferShape, flattenArray } from './shape';

export interface InitOptions {
  shape?: (number | null)[];
  dtype?: DType;
  device?: Device;
  requiresGrad?: boolean;
}

export function tensor(data: NestedArray<number>, options?: InitOptions): Tensor {
  const extractedShape = inferShape(data);
  const shape = options?.shape ?? (extractedShape.length > 0 ? extractedShape : [1]);

  const flattened = flattenArray(data);
  let buffer: ArrayBufferView;
  const dtype = options?.dtype ?? 'float32';
  if (dtype === 'float32') {
    buffer = new Float32Array(flattened);
  } else if (dtype === 'int32') {
    buffer = new Int32Array(flattened);
  } else {
    buffer = new Uint8Array(flattened);
  }

  return new Tensor({
    shape,
    dtype: options?.dtype,
    device: options?.device,
    requiresGrad: options?.requiresGrad,
    ctx: {
      op: 'Constant',
      inputs: [],
      attributes: {
        data: buffer,
      },
    },
  });
}

export function zeros(shape: (number | null)[], options?: InitOptions): Tensor {
  return new Tensor({
    shape,
    dtype: options?.dtype,
    device: options?.device,
    requiresGrad: options?.requiresGrad,
    ctx: {
      op: 'Zeros',
      inputs: [],
    },
  });
}

export function ones(shape: (number | null)[], options?: InitOptions): Tensor {
  return new Tensor({
    shape,
    dtype: options?.dtype,
    device: options?.device,
    requiresGrad: options?.requiresGrad,
    ctx: {
      op: 'Ones',
      inputs: [],
    },
  });
}
