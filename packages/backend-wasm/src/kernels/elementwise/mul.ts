import { mul, div } from '../../bindings';

export function executeMul(a: Float32Array, b: Float32Array, out: Float32Array): void {
  mul(a, b, out);
}

export function executeDiv(a: Float32Array, b: Float32Array, out: Float32Array): void {
  div(a, b, out);
}
