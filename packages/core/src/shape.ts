import { NestedArray } from './types';

export function inferShape(arr: NestedArray<number>): number[] {
  if (!Array.isArray(arr)) return [];
  const shape: number[] = [];
  let current: NestedArray<number> = arr;
  while (Array.isArray(current)) {
    shape.push(current.length);
    current = current[0]!;
  }
  return shape;
}

export function flattenArray(arr: NestedArray<number>): number[] {
  if (!Array.isArray(arr)) return [arr];
  const out: number[] = [];
  function recurse(a: NestedArray<number>[]) {
    for (const item of a) {
      if (Array.isArray(item)) {
        recurse(item);
      } else {
        out.push(item);
      }
    }
  }
  recurse(arr);
  return out;
}

export function broadcastShapes(
  shapeA: (number | null)[],
  shapeB: (number | null)[],
): (number | null)[] {
  const result: (number | null)[] = [];
  const maxRank = Math.max(shapeA.length, shapeB.length);

  const padA = maxRank - shapeA.length;
  const padB = maxRank - shapeB.length;

  for (let i = 0; i < maxRank; i++) {
    const dimA = i < padA ? 1 : shapeA[i - padA];
    const dimB = i < padB ? 1 : shapeB[i - padB];

    if (dimA === null || dimB === null) {
      if (dimA === 1) {
        result.push(dimB);
      } else if (dimB === 1) {
        result.push(dimA);
      } else {
        result.push(null);
      }
    } else if (dimA === 1) {
      result.push(dimB);
    } else if (dimB === 1) {
      result.push(dimA);
    } else if (dimA === dimB) {
      result.push(dimA);
    } else {
      throw new Error(`Cannot broadcast shapes [${shapeA.join(', ')}] and [${shapeB.join(', ')}]`);
    }
  }

  return result;
}

export function shapeSize(shape: (number | null)[]): number | null {
  let size = 1;
  for (const dim of shape) {
    if (dim === null) return null;
    size *= dim;
  }
  return size;
}

export function shapesEqual(shapeA: (number | null)[], shapeB: (number | null)[]): boolean {
  if (shapeA.length !== shapeB.length) return false;
  for (let i = 0; i < shapeA.length; i++) {
    if (shapeA[i] !== shapeB[i]) return false;
  }
  return true;
}

/**
 * Compute output shape of a reduce operation.
 * axes must be sorted non-negative indices into inputShape.
 * keepdim=true: reduced dims remain as size 1; keepdim=false: they are removed.
 */
export function reduceOutputShape(
  inputShape: (number | null)[],
  axes: number[],
  keepdim: boolean,
): number[] {
  const axisSet = new Set(axes);
  const out: number[] = [];
  for (let i = 0; i < inputShape.length; i++) {
    if (axisSet.has(i)) {
      if (keepdim) out.push(1);
    } else {
      out.push(inputShape[i] as number);
    }
  }
  return out; // full reduce + keepdim=false → rank-0 ([]); PyTorch semantics.
}

/**
 * Resolve a reshape target shape that may contain a single `null` placeholder
 * (PyTorch's `-1`) by dividing the input size by the product of the other dims.
 * Throws if more than one null is present, the input size is unknown, or the
 * target size is incompatible with the input size.
 */
export function resolveShapeInference(
  inputShape: (number | null)[],
  targetShape: (number | null)[],
): number[] {
  const nullIndices: number[] = [];
  let product = 1;
  for (let i = 0; i < targetShape.length; i++) {
    const d = targetShape[i];
    if (d === null) {
      nullIndices.push(i);
    } else {
      if (!Number.isInteger(d) || d < 0) {
        throw new Error(`reshape: invalid dim ${d} at index ${i}`);
      }
      product *= d;
    }
  }

  const inputSize = shapeSize(inputShape);

  if (nullIndices.length === 0) {
    if (inputSize !== null && inputSize !== product) {
      throw new Error(
        `reshape: cannot reshape tensor of size ${inputSize} into shape [${targetShape.join(', ')}]`,
      );
    }
    return targetShape as number[];
  }

  if (nullIndices.length > 1) {
    throw new Error(`reshape: only one dimension can be inferred (got ${nullIndices.length} nulls)`);
  }

  if (inputSize === null) {
    throw new Error(`reshape: cannot infer dim from input shape with dynamic dims`);
  }

  if (product === 0) {
    throw new Error(`reshape: cannot infer dim when other dims have zero product`);
  }

  if (inputSize % product !== 0) {
    throw new Error(
      `reshape: cannot reshape tensor of size ${inputSize} into shape [${targetShape.join(', ')}]`,
    );
  }

  const inferred = inputSize / product;
  const out = targetShape.slice() as number[];
  out[nullIndices[0]!] = inferred;
  return out;
}

/**
 * Normalise an axes argument (number | number[] | undefined) to a sorted
 * non-negative array of axis indices against a tensor of the given rank.
 */
export function normalizeAxes(axes: number | number[] | undefined, rank: number): number[] {
  const raw =
    axes === undefined
      ? Array.from({ length: rank }, (_, i) => i)
      : Array.isArray(axes)
        ? axes
        : [axes];
  return raw.map((ax) => (ax < 0 ? rank + ax : ax)).sort((a, b) => a - b);
}
