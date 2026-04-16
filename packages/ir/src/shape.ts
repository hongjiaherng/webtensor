/**
 * Compute C-order (row-major) strides for the given concrete shape.
 * The innermost dimension has stride 1; each outer dimension has stride equal
 * to the product of all inner dimensions.
 *
 * Example: shape [2, 3, 4] → strides [12, 4, 1]
 */
export function computeContiguousStrides(shape: number[]): number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}
