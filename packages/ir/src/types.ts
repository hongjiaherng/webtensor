export type DType = 'float32' | 'int32' | 'bool';

/**
 * Maximum tensor rank supported across all backends. PyTorch-aligned.
 * All kernel meta layouts (WASM, WebGPU) size their fixed shape/strides arrays
 * to this bound; bumping it requires rebuilding the WASM package and updating
 * the WGSL `__TENSOR_META__` struct (via `@webtensor/backend-webgpu` utils).
 */
export const MAX_RANK = 64;

export type AttributeValue =
  | number
  | string
  | number[]
  | string[]
  | boolean
  | ArrayBuffer
  | ArrayBufferView;

export interface Node {
  id: string;
  op: string; // e.g. "Add", "MatMul", etc.
  inputs: string[]; // Value names
  outputs: string[]; // Value names
  attributes?: Record<string, AttributeValue>;
  name?: string;
}

export interface Value {
  name: string;
  shape: (number | null)[];
  dtype: DType;
  data?: ArrayBuffer; // only for weights/constants
  producer?: string; // Node ID
  consumers?: string[]; // Node IDs
  debugName?: string;
}

export interface Graph {
  nodes: Node[];
  values: Record<string, Value>;
  inputs: string[];
  outputs: string[];
  initializers: string[];
  name?: string;
  opset?: number;
}
