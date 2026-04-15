import { Tensor } from './tensor';
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
  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad || b.requiresGrad,
    ctx: { op: 'Sub', inputs: [a, b] },
  });
}

export function div(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: a.requiresGrad || b.requiresGrad,
    ctx: { op: 'Div', inputs: [a, b] },
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
    requiresGrad: false,
    ctx: { op: 'Reshape', inputs: [a], attributes: { shape } },
  });
}

export function slice(a: Tensor, starts: number[], ends: number[]): Tensor {
  if (starts.length !== a.shape.length || ends.length !== a.shape.length) {
    throw new Error('slice: starts and ends must have the same length as the tensor rank');
  }
  const outShape = starts.map((s, i) => ends[i] - s);

  return new Tensor({
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: false,
    ctx: { op: 'Slice', inputs: [a], attributes: { starts, ends } },
  });
}

export function contiguous(a: Tensor): Tensor {
  return new Tensor({
    shape: a.shape,
    dtype: a.dtype,
    device: a.device,
    requiresGrad: false,
    ctx: { op: 'Contiguous', inputs: [a] },
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
