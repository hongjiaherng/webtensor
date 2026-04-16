// webtensor/packages/core/src/types.ts

export type Device = 'cpu' | 'wasm' | 'webgpu';

export interface OpContext<T = any> {
  op: string; // "Add", "MatMul", etc.
  inputs: T[];
  // reverse-mode diff closure: computes gradients for inputs given the output gradient
  backward?: (grad: T) => T[];
  attributes?: Record<string, any>;
}

export type NestedArray<T> = T | NestedArray<T>[];
