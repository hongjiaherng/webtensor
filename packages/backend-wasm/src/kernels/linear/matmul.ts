import { matmul } from '../../bindings';

export function executeMatMul(
  a: Float32Array, b: Float32Array, out: Float32Array,
  m: number, k: number, n: number
): void {
  matmul(a, b, out, m, k, n);
}
