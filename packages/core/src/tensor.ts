import { DType, Device, OpContext } from './types';
import { add } from './ops';
import { computeStrides, shapeSize } from './shape';

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

  constructor(options: TensorOptions) {
    this.id = `t_${tensorIdCounter++}`;
    this.shape = options.shape;
    
    const concreteShape = this.shape.map(s => s ?? 1) as number[];
    this.strides = options.strides ?? computeStrides(concreteShape);
    this.size = shapeSize(concreteShape) ?? 0;
    
    this.dtype = options.dtype ?? 'float32';
    this.device = options.device ?? 'cpu';
    this.requiresGrad = options.requiresGrad ?? false;
    this._ctx = options.ctx;
  }

  backward() {
    if (!this.requiresGrad) {
      throw new Error("Cannot call backward() on a tensor that does not require gradients.");
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
          attributes: { data: gradData }
        }
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

  // --- Convenience Methods to enable method chaining ---
  
  add(other: Tensor): Tensor {
    return add(this, other);
  }
}
