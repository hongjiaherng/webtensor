import { add, sub } from '../../bindings';

export function executeAdd(a: Float32Array, b: Float32Array, out: Float32Array): void {
  add(a, b, out);
}

export function executeSub(a: Float32Array, b: Float32Array, out: Float32Array): void {
  sub(a, b, out);
}
