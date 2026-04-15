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
  dtype: 'float32' | 'int32' | 'bool';
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
