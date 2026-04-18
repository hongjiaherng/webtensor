import { DType } from '@webtensor/ir';
import { TypedArray } from '@webtensor/runtime';
import { Device, OpContext } from './types';
import {
  add,
  sub,
  mul,
  div,
  matmul,
  transpose,
  reshape,
  slice,
  contiguous,
  view,
  unsqueeze,
  squeeze,
  permute,
  flatten,
  expand,
  neg,
  exp,
  log,
  sqrt,
  abs,
  pow,
  clone,
  detach,
  sum,
  mean,
} from './ops';
import { computeContiguousStrides } from '@webtensor/ir';
import { shapeSize } from './shape';
import { equal, allclose, AllcloseOptions } from './compare';
import { run, RunOptions } from './run';

let tensorIdCounter = 0;

export interface TensorOptions {
  shape: (number | null)[];
  dtype?: DType;
  device?: Device;
  requiresGrad?: boolean;
  ctx?: OpContext<Tensor>;
  strides?: number[];
}

export class Tensor {
  id: string;
  shape: (number | null)[];
  strides: number[];
  size: number;

  dtype: DType;
  device: Device;

  requiresGrad: boolean;
  grad?: Tensor;
  _ctx?: OpContext<Tensor>;

  /**
   * For trainable parameters (created via factory functions with `requiresGrad: true`):
   * the raw data that `compile()` feeds into the graph each call and that the
   * optimizer mutates in place via `opt.step()`. Users rarely touch this directly.
   * Not set for constants or for tensors that are results of ops.
   */
  data?: TypedArray;

  constructor(options: TensorOptions) {
    this.id = `t_${tensorIdCounter++}`;
    this.shape = options.shape;

    const concreteShape = this.shape.map((s) => s ?? 1) as number[];
    this.strides = options.strides ?? computeContiguousStrides(concreteShape);
    this.size = shapeSize(concreteShape) ?? 0;

    this.dtype = options.dtype ?? 'float32';
    this.device = options.device ?? 'cpu';
    this.requiresGrad = options.requiresGrad ?? false;
    this._ctx = options.ctx;
  }

  backward() {
    if (!this.requiresGrad) {
      throw new Error('Cannot call backward() on a tensor that does not require gradients.');
    }

    const topo: Tensor[] = [];
    const visited = new Set<string>();

    const buildTopo = (t: Tensor) => {
      if (visited.has(t.id)) return;
      visited.add(t.id);
      if (t._ctx) {
        for (const input of t._ctx.inputs) {
          if (input instanceof Tensor) {
            buildTopo(input);
          }
        }
      }
      topo.push(t);
    };

    buildTopo(this);

    // Initialize gradient of this tensor to a dummy tensor representing 1s.
    // The runtime is natively responsible for populating this with actual 1.0 values.
    if (!this.grad) {
      const gradData = new Float32Array(this.size).fill(1.0);
      this.grad = new Tensor({
        shape: this.shape,
        dtype: this.dtype,
        device: this.device,
        requiresGrad: false,
        ctx: {
          op: 'Constant',
          inputs: [],
          attributes: { data: gradData },
        },
      });
    }

    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      if (!t._ctx || !t._ctx.backward || !t.grad) continue;

      const inputGrads = t._ctx.backward(t.grad);

      for (let j = 0; j < t._ctx.inputs.length; j++) {
        const input = t._ctx.inputs[j] as Tensor;
        if (input.requiresGrad) {
          const g = inputGrads[j];
          if (!input.grad) {
            input.grad = g;
          } else {
            // Accumulate gradients: input.grad += g
            input.grad = add(input.grad, g);
          }
        }
      }
    }
  }

  // --- Accessor Methods (PyTorch API parity) ---

  dim(): number {
    return this.shape.length;
  }

  numel(): number {
    return this.size;
  }

  isContiguous(): boolean {
    const concreteShape = this.shape.map((s) => s ?? 1) as number[];
    const expected = computeContiguousStrides(concreteShape);
    if (this.strides.length !== expected.length) return false;
    for (let i = 0; i < expected.length; i++) {
      if (this.strides[i] !== expected[i]) return false;
    }
    return true;
  }

  stride(): number[] {
    return this.strides;
  }

  // --- Convenience Methods to enable method chaining ---

  add(other: Tensor): Tensor {
    return add(this, other);
  }
  sub(other: Tensor): Tensor {
    return sub(this, other);
  }
  mul(other: Tensor): Tensor {
    return mul(this, other);
  }
  div(other: Tensor): Tensor {
    return div(this, other);
  }
  matmul(other: Tensor): Tensor {
    return matmul(this, other);
  }
  transpose(): Tensor {
    return transpose(this);
  }
  reshape(shape: number[]): Tensor {
    return reshape(this, shape);
  }
  slice(starts: number[], ends: number[]): Tensor {
    return slice(this, starts, ends);
  }
  contiguous(): Tensor {
    return contiguous(this);
  }
  view(shape: number[]): Tensor {
    return view(this, shape);
  }
  unsqueeze(dim: number): Tensor {
    return unsqueeze(this, dim);
  }
  squeeze(dim?: number): Tensor {
    return squeeze(this, dim);
  }
  permute(axes: number[]): Tensor {
    return permute(this, axes);
  }
  flatten(startDim: number = 0, endDim: number = -1): Tensor {
    return flatten(this, startDim, endDim);
  }
  expand(shape: number[]): Tensor {
    return expand(this, shape);
  }
  neg(): Tensor {
    return neg(this);
  }
  exp(): Tensor {
    return exp(this);
  }
  log(): Tensor {
    return log(this);
  }
  sqrt(): Tensor {
    return sqrt(this);
  }
  abs(): Tensor {
    return abs(this);
  }
  pow(exponent: number): Tensor {
    return pow(this, exponent);
  }
  clone(): Tensor {
    return clone(this);
  }
  detach(): Tensor {
    return detach(this);
  }
  sum(axis?: number | number[], keepdim: boolean = false): Tensor {
    return sum(this, axis, keepdim);
  }
  mean(axis?: number | number[], keepdim: boolean = false): Tensor {
    return mean(this, axis, keepdim);
  }
  zeroGrad(): void {
    this.grad = undefined;
  }

  /**
   * Eagerly evaluate this tensor and return a new `Tensor` with `.data`
   * populated. Equivalent to `run(this, options)` — see `./run.ts`.
   *
   * ```ts
   * const y = await add(tensor([1, 2]), tensor([3, 4])).run();
   * console.log(y.data);   // Float32Array [4, 6]
   * ```
   */
  run(options?: RunOptions): Promise<Tensor> {
    return run(this, options);
  }

  /** Strict equality (same shape + exact values). See `compare.equal`. */
  equals(other: Tensor): boolean {
    return equal(this, other);
  }

  /** Numeric closeness (same shape + values within tolerances). See `compare.allclose`. */
  allclose(other: Tensor, opts?: AllcloseOptions): boolean {
    return allclose(this, other, opts);
  }
}
