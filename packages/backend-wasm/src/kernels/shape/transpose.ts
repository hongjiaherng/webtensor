import { transpose } from '../../bindings';

export function executeTranspose(
  a: Float32Array, out: Float32Array,
  m: number, n: number
): void {
  transpose(a, out, m, n);
}
