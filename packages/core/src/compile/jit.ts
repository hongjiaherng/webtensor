import { DType } from '@webtensor/ir';
import { Engine, TypedArray } from '@webtensor/runtime';
import { Tensor } from '../tensor';
import { placeholder } from '../ops';
import { compileGraph } from './trace';
import { Device } from '../types';

// ---------------------------------------------------------------------------
// compile() — trace a TypeScript function once, return a callable that feeds
// data through the compiled graph on each invocation.
// Users never see `placeholder()` IDs, `compileGraph`, or `engine.get`.

export type ShapeLike = number[] | { shape: number[]; dtype?: DType };

/**
 * Anything a user can reasonably use as feed data for a compiled function.
 * We accept `Tensor` (the usual case), or a raw `TypedArray` (for fast paths
 * with data loaders), or a nested array (for ad-hoc testing).
 */
export type FeedValue = Tensor | TypedArray | number[];

function normalizeShape(s: ShapeLike): { shape: number[]; dtype: DType } {
  if (Array.isArray(s)) return { shape: s, dtype: 'float32' };
  return { shape: s.shape, dtype: s.dtype ?? 'float32' };
}

function toArrayBufferView(v: FeedValue, dtype: DType): ArrayBufferView {
  if (v instanceof Tensor) {
    if (!v.data) {
      throw new Error(
        'compile feed: passed a Tensor without .data — only parameters ' +
          '(requiresGrad: true) or tensors produced by a prior compile() call carry .data. ' +
          'Use `tensor([...])` or pass a TypedArray directly.',
      );
    }
    return v.data;
  }
  if (ArrayBuffer.isView(v)) return v;
  // Nested array → allocate a typed array
  if (dtype === 'float32') return new Float32Array(v as number[]);
  if (dtype === 'int32') return new Int32Array(v as number[]);
  return Uint8Array.from(v as number[]);
}

export interface CompileOptions {
  /** Device to execute on. Ignored if `engine` is supplied. */
  device?: Device;
  /** Pre-existing engine to reuse. Overrides `device`. */
  engine?: Engine;
}

type InputsFromSpec<S> = { [K in keyof S]: Tensor };
type FeedsFromSpec<S> = { [K in keyof S]: FeedValue };

type OutputShape = Tensor | Tensor[] | Record<string, Tensor>;

/**
 * The compiled function returns the same shape the user returned from the
 * traced function, but with every `Tensor` replaced by an evaluated `Tensor`
 * whose `.data` holds the result.
 */
type CompiledOutput<O> = O extends Tensor
  ? Tensor
  : O extends Tensor[]
    ? Tensor[]
    : O extends Record<string, Tensor>
      ? { [K in keyof O]: Tensor }
      : never;

/**
 * Compile a TypeScript function into a callable that accepts feed data and
 * returns evaluated tensors. The function's arguments are traced once as
 * placeholders; subsequent calls feed concrete data through the same graph.
 *
 * Example — forward only:
 * ```ts
 * const forward = await compile(
 *   ({ x, W, b }) => relu(add(matmul(x, W), b)),
 *   { x: [4, 2], W: [2, 8], b: [8] },
 * );
 * const y = await forward({ x: xData, W: w1, b: b1 });   // y is a Tensor
 * console.log(y.data);   // Float32Array of the result
 * ```
 *
 * Example — training (parameters declared with `requiresGrad: true` outside;
 * compile auto-feeds them on every call):
 * ```ts
 * const W = randn([2, 8], { requiresGrad: true });
 * const b = zeros([8],    { requiresGrad: true });
 *
 * const step = await compile(
 *   ({ x, y }) => {
 *     const loss = mseLoss(add(matmul(x, W), b), y);
 *     return { loss, dW: grad(loss, W), db: grad(loss, b) };
 *   },
 *   { x: [4, 2], y: [4, 1] },
 * );
 *
 * const opt = new SGD(0.1);
 * for (let i = 0; i < 1000; i++) {
 *   const { loss, dW, db } = await step({ x: xBatch, y: yBatch });
 *   opt.step([W, b], [dW, db]);
 * }
 * ```
 *
 * @category Compile
 */
export async function compile<S extends Record<string, ShapeLike>, O extends OutputShape>(
  fn: (inputs: InputsFromSpec<S>) => O,
  spec: S,
  options: CompileOptions = {},
): Promise<(data: FeedsFromSpec<S>) => Promise<CompiledOutput<O>>> {
  const engine = options.engine ?? (await Engine.create(options.device ?? 'cpu'));
  const device = options.device ?? 'cpu';

  // Build placeholders and stash their IDs + dtypes by name.
  const inputPlaceholders: Record<string, Tensor> = {};
  const inputIdByName: Record<string, string> = {};
  const inputDtypeByName: Record<string, DType> = {};
  for (const key of Object.keys(spec)) {
    const { shape, dtype } = normalizeShape(spec[key]);
    const p = placeholder(shape, dtype, device);
    inputPlaceholders[key] = p;
    inputIdByName[key] = p.id;
    inputDtypeByName[key] = dtype;
  }

  const output = fn(inputPlaceholders as InputsFromSpec<S>);

  // Classify output shape: single Tensor, Tensor[], or named record.
  let mode: 'single' | 'array' | 'record';
  let outputTensors: Tensor[];
  let outputKeys: string[] = [];
  if (output instanceof Tensor) {
    mode = 'single';
    outputTensors = [output];
  } else if (Array.isArray(output)) {
    mode = 'array';
    outputTensors = output;
  } else {
    mode = 'record';
    outputKeys = Object.keys(output);
    outputTensors = outputKeys.map((k) => (output as Record<string, Tensor>)[k]);
  }

  // Walk the traced graph and collect trainable parameters: any reachable
  // Tensor whose op is 'Placeholder' AND that carries attached `.data`.
  // These are auto-fed on every call from their current `.data` contents.
  const params = collectParams(outputTensors);

  const graph = compileGraph(outputTensors);

  return async (data: FeedsFromSpec<S>): Promise<CompiledOutput<O>> => {
    for (const key of Object.keys(spec)) {
      if (!(key in data)) {
        throw new Error(`compile: missing feed for '${key}'`);
      }
    }
    const feeds: Record<string, ArrayBufferView> = {};
    for (const key of Object.keys(spec)) {
      feeds[inputIdByName[key]] = toArrayBufferView(
        (data as Record<string, FeedValue>)[key],
        inputDtypeByName[key],
      );
    }
    // Auto-feed every trainable parameter from its current `.data`.
    for (const p of params) {
      feeds[p.id] = p.data!;
    }

    await engine.evaluate(graph, feeds);

    // Allocate a fresh wrapper Tensor for each output this call — users expect
    // independent Tensor objects so logging / stashing results is safe.
    const wrappers: Tensor[] = [];
    for (const t of outputTensors) {
      const w = new Tensor({
        shape: t.shape,
        dtype: t.dtype,
        device: t.device,
        requiresGrad: false,
      });
      w.data = (await engine.get(t.id)) as TypedArray;
      wrappers.push(w);
    }

    if (mode === 'single') return wrappers[0] as CompiledOutput<O>;
    if (mode === 'array') return wrappers as CompiledOutput<O>;
    const out: Record<string, Tensor> = {};
    for (let i = 0; i < outputKeys.length; i++) {
      out[outputKeys[i]] = wrappers[i];
    }
    return out as CompiledOutput<O>;
  };
}

// ---------------------------------------------------------------------------
// Walk a tensor graph and collect all trainable parameters:
// Placeholder tensors with attached `.data` (created via factories with
// `requiresGrad: true`). These are auto-fed on every compile call.

function collectParams(outputs: Tensor[]): Tensor[] {
  const visited = new Set<string>();
  const params: Tensor[] = [];
  const walk = (t: Tensor) => {
    if (visited.has(t.id)) return;
    visited.add(t.id);
    if (t._ctx?.op === 'Placeholder' && t.data !== undefined) {
      params.push(t);
      return; // params have no inputs
    }
    if (t._ctx) {
      for (const input of t._ctx.inputs) {
        if (input instanceof Tensor) walk(input);
      }
    }
  };
  for (const t of outputs) walk(t);
  return params;
}
