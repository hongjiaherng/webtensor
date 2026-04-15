export function getShapeSize(shape: (number | null)[]): number {
  let size = 1;
  for (const dim of shape) {
    if (dim === null) throw new Error("Dynamic dimensions not yet supported in WebGPU backend");
    size *= dim;
  }
  return size;
}

export function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}
