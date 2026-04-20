import { Engine } from '@webtensor/runtime';
import { Tensor } from './tensor';
import { compileGraph } from './compile/trace';
import { Device } from './types';

export interface RunOptions {
  /** Device to execute on. Defaults to 'cpu'. Ignored if `engine` is supplied. */
  device?: Device;
  /** Pre-existing engine to reuse across multiple `run()` calls. */
  engine?: Engine;
}

/**
 * Evaluate a `Tensor` (or multiple tensors) eagerly and return the result(s)
 * with `.data` populated. Convenience wrapper around
 * `compileGraph(...)` + `engine.evaluate(graph)` + `engine.get(id)` for
 * one-shot use — the common case in tests and simple examples.
 *
 * For reusable training loops prefer `compile(fn, spec)` so the graph is
 * traced only once.
 *
 * @example
 * ```ts
 * import { tensor, add, mul, run } from '@webtensor/core';
 *
 * const y = await run(add(tensor([1, 2, 3]), tensor([4, 5, 6])));
 * console.log(y.data); // Float32Array [5, 7, 9]
 *
 * // multiple outputs in one pass
 * const [yA, yB] = await run([add(a, b), mul(a, b)], { device: 'webgpu' });
 * ```
 *
 * @category Compile
 */
export async function run(t: Tensor, options?: RunOptions): Promise<Tensor>;
export async function run(ts: Tensor[], options?: RunOptions): Promise<Tensor[]>;
export async function run(
  input: Tensor | Tensor[],
  options: RunOptions = {},
): Promise<Tensor | Tensor[]> {
  const tensors = Array.isArray(input) ? input : [input];
  const engine = options.engine ?? (await Engine.create(options.device ?? 'cpu'));

  const graph = compileGraph(tensors);
  await engine.evaluate(graph);

  const results: Tensor[] = [];
  for (const t of tensors) {
    const data = await engine.get(t.id);
    const wrap = new Tensor({
      shape: t.shape,
      dtype: t.dtype,
      device: t.device,
      requiresGrad: false,
    });
    wrap.data = data as Tensor['data'];
    results.push(wrap);
  }
  return Array.isArray(input) ? results : results[0];
}
