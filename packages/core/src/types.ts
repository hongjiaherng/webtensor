// webtensor/packages/core/src/types.ts

import type { AttributeValue } from '@webtensor/ir';

export type Device = 'cpu' | 'wasm' | 'webgpu';

export interface OpContext<T = unknown> {
  op: string; // "Add", "MatMul", etc.
  inputs: T[];
  // reverse-mode diff closure: computes gradients for inputs given the output gradient
  backward?: (grad: T) => T[];
  attributes?: Record<string, AttributeValue>;
}

export type NestedArray<T> = T | NestedArray<T>[];
