import { Tensor } from './tensor';
import { tensor } from './tensor_init';
import { broadcastShapes } from './shape';

export function add(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const requiresGrad = a.requiresGrad || b.requiresGrad;

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad,
    ctx: {
      op: 'Add',
      inputs: [a, b],
      backward: (grad: Tensor) => {
        const gradA = grad;
        const gradB = grad;

        // NOTE: In a full implementation, if `shapesEqual(a.shape, outShape)` is false,
        // we must insert a `reduceSum` operator to unbroadcast the gradients.

        return [gradA, gradB];
      },
    },
  });
}

export function sub(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const requiresGrad = a.requiresGrad || b.requiresGrad;
  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad,
    ctx: {
      op: 'Sub',
      inputs: [a, b],
      backward: (grad: Tensor) => {
        const gradA = grad;
        const gradB = mul(grad, tensor([-1]));
        // NOTE: missing unbroadcast — requires ReduceSum op (Phase 11)
        return [gradA, gradB];
      },
    },
  });
}

export function div(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const requiresGrad = a.requiresGrad || b.requiresGrad;
  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad,
    ctx: {
      op: 'Div',
      inputs: [a, b],
      backward: (grad: Tensor) => {
        // d(a/b)/da = 1/b → gradA = grad / b
        const gradA = div(grad, b);
        // d(a/b)/db = -a/b^2 → gradB = -grad * a / (b * b)
        const gradB = mul(div(mul(grad, a), mul(b, b)), tensor([-1]));
        // NOTE: missing unbroadcast — requires ReduceSum op (Phase 11)
        return [gradA, gradB];
      },
    },
  });
}

export function mul(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const requiresGrad = a.requiresGrad || b.requiresGrad;

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad,
    ctx: {
      op: 'Mul',
      inputs: [a, b],
      backward: (grad: Tensor) => {
        // Backward pass of mul constructs new nodes in the graph
        const gradA = mul(grad, b);
        const gradB = mul(grad, a);

        return [gradA, gradB];
      },
    },
  });
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  // Simplistic shape check: a=[..., M, K], b=[..., K, N] => [..., M, N]
  if (a.shape.length < 2 || b.shape.length < 2) {
    throw new Error('MatMul requires inputs to be at least 2T');
  }
  const K_A = a.shape[a.shape.length - 1];
  const K_B = b.shape[b.shape.length - 2];

  if (K_A !== null && K_B !== null && K_A !== K_B) {
    throw new Error(`MatMul inner dimensions must match: ${K_A} != ${K_B}`);
  }

  const outShape = [...a.shape.slice(0, -2)];
  // NOTE: Proper MatMul broadcasting for batch dimensions goes here
  outShape.push(a.shape[a.shape.length - 2]);
  outShape.push(b.shape[b.shape.length - 1]);

  const requiresGrad = a.requiresGrad || b.requiresGrad;

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad,
    ctx: {
      op: 'MatMul',
      inputs: [a, b],
      backward: (grad: Tensor) => {
        // A*B = C
        // dC/dA = grad * B^T
        // dC/dB = A^T * grad
        const gradA = matmul(grad, transpose(b));
        const gradB = matmul(transpose(a), grad);
        return [gradA, gradB];
      },
    },
  });
}

export function transpose(a: Tensor): Tensor {
  if (a.shape.length < 2) {
    throw new Error('Transpose requires at least 2 dimensions');
  }
  const outShape = [...a.shape];
  const last = outShape.length - 1;
  [outShape[last - 1], outShape[last]] = [outShape[last], outShape[last - 1]];

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Transpose',
      inputs: [a],
      backward: (grad: Tensor) => {
        return [transpose(grad)];
      },
    },
  });
}

export function reshape(a: Tensor, shape: number[]): Tensor {
  return new Tensor({
    shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Reshape',
      inputs: [a],
      attributes: { shape },
      backward: (grad: Tensor) => {
        return [reshape(grad, a.shape as number[])];
      },
    },
  });
}

export function slice(a: Tensor, starts: number[], ends: number[]): Tensor {
  if (starts.length !== a.shape.length || ends.length !== a.shape.length) {
    throw new Error('slice: starts and ends must have the same length as the tensor rank');
  }
  for (let i = 0; i < starts.length; i++) {
    const dim = a.shape[i];
    if (starts[i] < 0 || (dim !== null && ends[i] > dim) || starts[i] >= ends[i]) {
      throw new Error(`slice: invalid range [${starts[i]}, ${ends[i]}) for dim ${i} (size ${dim})`);
    }
  }
  const outShape = starts.map((s, i) => ends[i] - s);

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Slice',
      inputs: [a],
      attributes: { starts, ends },
      // NOTE: slice backward requires a Pad/Scatter op to place gradients back — not yet implemented
    },
  });
}

export function contiguous(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Contiguous',
      inputs: [a],
      backward: (grad: Tensor) => {
        return [grad];
      },
    },
  });
}

export function view(a: Tensor, shape: number[]): Tensor {
  return new Tensor({
    shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'View',
      inputs: [a],
      attributes: { shape },
      backward: (grad: Tensor) => {
        return [view(grad, a.shape as number[])];
      },
    },
  });
}

export function unsqueeze(a: Tensor, dim: number): Tensor {
  const rank = a.shape.length;
  const d = dim < 0 ? rank + 1 + dim : dim;
  if (d < 0 || d > rank) {
    throw new Error(`unsqueeze: dim ${dim} out of range for tensor of rank ${rank}`);
  }
  const outShape = [...a.shape];
  outShape.splice(d, 0, 1);

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Unsqueeze',
      inputs: [a],
      attributes: { dim: d },
      backward: (grad: Tensor) => {
        return [squeeze(grad, d)];
      },
    },
  });
}

export function squeeze(a: Tensor, dim?: number): Tensor {
  let outShape: (number | null)[];
  if (dim !== undefined) {
    const d = dim < 0 ? a.shape.length + dim : dim;
    if (d < 0 || d >= a.shape.length) {
      throw new Error(`squeeze: dim ${dim} out of range for tensor of rank ${a.shape.length}`);
    }
    if (a.shape[d] !== 1) {
      throw new Error(`squeeze: dimension ${d} has size ${a.shape[d]}, expected 1`);
    }
    outShape = [...a.shape];
    outShape.splice(d, 1);
  } else {
    outShape = a.shape.filter((s) => s !== 1);
    if (outShape.length === 0) outShape = [1];
  }

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Squeeze',
      inputs: [a],
      attributes: dim !== undefined ? { dim } : undefined,
      backward: (grad: Tensor) => {
        return [reshape(grad, a.shape as number[])];
      },
    },
  });
}

export function permute(a: Tensor, axes: number[]): Tensor {
  if (axes.length !== a.shape.length) {
    throw new Error(`permute: axes length ${axes.length} must match tensor rank ${a.shape.length}`);
  }
  const outShape = axes.map((ax) => a.shape[ax]);

  // Compute inverse permutation for backward
  const inverseAxes = new Array(axes.length);
  for (let i = 0; i < axes.length; i++) {
    inverseAxes[axes[i]] = i;
  }

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Permute',
      inputs: [a],
      attributes: { axes },
      backward: (grad: Tensor) => {
        return [permute(grad, inverseAxes)];
      },
    },
  });
}

export function flatten(a: Tensor, startDim: number = 0, endDim: number = -1): Tensor {
  const rank = a.shape.length;
  const sd = startDim < 0 ? rank + startDim : startDim;
  const ed = endDim < 0 ? rank + endDim : endDim;

  if (sd < 0 || sd >= rank || ed < 0 || ed >= rank || sd > ed) {
    throw new Error(`flatten: invalid dims [${startDim}, ${endDim}] for rank ${rank}`);
  }

  const before = a.shape.slice(0, sd) as number[];
  const middle = a.shape.slice(sd, ed + 1) as number[];
  const after = a.shape.slice(ed + 1) as number[];
  const flatDim = middle.reduce((acc, d) => acc * (d ?? 1), 1);

  const newShape = [...before, flatDim, ...after];
  return reshape(a, newShape);
}

export function expand(a: Tensor, shape: number[]): Tensor {
  if (shape.length < a.shape.length) {
    throw new Error(
      `expand: target shape rank ${shape.length} must be >= tensor rank ${a.shape.length}`,
    );
  }
  // Validate: each dim must either match or be 1 in the source
  const rankOffset = shape.length - a.shape.length;
  for (let i = 0; i < a.shape.length; i++) {
    const srcDim = a.shape[i];
    const tgtDim = shape[rankOffset + i];
    if (srcDim !== 1 && srcDim !== tgtDim) {
      throw new Error(
        `expand: cannot expand dim ${i} from ${srcDim} to ${tgtDim} (must be 1 or match)`,
      );
    }
  }

  return new Tensor({
    shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Expand',
      inputs: [a],
      attributes: { shape },
      // NOTE: expand backward requires sum reduction over expanded dims — not yet implemented
    },
  });
}

export function relu(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Relu',
      inputs: [a],
      backward: (grad: Tensor) => {
        const gradA = new Tensor({
          shape: a.shape,
          dtype: a.dtype,
          device: a.device,
          requiresGrad: false,
          ctx: {
            op: 'ReluGrad',
            inputs: [grad, a],
          },
        });
        return [gradA];
      },
    },
  });
}

export function neg(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Neg',
      inputs: [a],
      backward: (grad: Tensor) => {
        return [neg(grad)];
      },
    },
  });
}

export function exp(a: Tensor): Tensor {
  const out = new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Exp',
      inputs: [a],
      backward: (grad: Tensor) => {
        // d/dx exp(x) = exp(x)
        return [mul(grad, exp(a))];
      },
    },
  });
  return out;
}

export function log(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Log',
      inputs: [a],
      backward: (grad: Tensor) => {
        // d/dx ln(x) = 1/x
        return [div(grad, a)];
      },
    },
  });
}

export function sqrt(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Sqrt',
      inputs: [a],
      backward: (grad: Tensor) => {
        // d/dx sqrt(x) = 1 / (2 * sqrt(x))
        return [div(grad, mul(tensor([2]), sqrt(a)))];
      },
    },
  });
}

export function abs(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Abs',
      inputs: [a],
      // NOTE: abs backward requires sign(a) — not yet implemented
    },
  });
}

export function pow(a: Tensor, exponent: number): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Pow',
      inputs: [a],
      attributes: { exponent },
      backward: (grad: Tensor) => {
        // d/dx x^n = n * x^(n-1)
        return [mul(mul(grad, tensor([exponent])), pow(a, exponent - 1))];
      },
    },
  });
}

export function sigmoid(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Sigmoid',
      inputs: [a],
      backward: (grad: Tensor) => {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        const s = sigmoid(a);
        return [mul(grad, mul(s, sub(tensor([1]), s)))];
      },
    },
  });
}

export function tanh(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Tanh',
      inputs: [a],
      backward: (grad: Tensor) => {
        // d/dx tanh(x) = 1 - tanh(x)^2
        const t = tanh(a);
        return [mul(grad, sub(tensor([1]), mul(t, t)))];
      },
    },
  });
}

export function clone(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad,
    ctx: {
      op: 'Contiguous',
      inputs: [a],
      backward: (grad: Tensor) => {
        return [grad];
      },
    },
  });
}

export function detach(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: false,
  });
}
