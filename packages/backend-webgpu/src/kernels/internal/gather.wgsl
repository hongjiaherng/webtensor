// gather.wgsl — copies a strided/offset source tensor into a packed contiguous
// destination buffer.
//
// meta layout (array<u32>, 19 elements = 76 bytes):
//   [0]      total elements in logical tensor
//   [1]      rank
//   [2..9]   shape[0..7]   (unused entries are 0)
//   [10..17] strides[0..7] (unused entries are 0)
//   [18]     src_offset (element offset into src, not bytes)
//
// Using storage read (not uniform) to avoid the 16-byte element padding rule
// that WGSL imposes on array<u32, N> inside uniform structs.

@group(0) @binding(0) var<storage, read>       src  : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst  : array<f32>;
@group(0) @binding(2) var<storage, read>       meta : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let flat  = gid.x;
  let total = meta[0];
  if (flat >= total) { return; }

  let rank       = meta[1];
  let src_offset = meta[18];

  // Decompose flat output index into per-axis coordinates, then dot with strides.
  var remaining = flat;
  var src_idx   = src_offset;
  for (var i = rank; i > 0u; i--) {
    let axis = i - 1u;
    let dim  = meta[2u  + axis];  // shape[axis]
    let s    = meta[10u + axis];  // strides[axis]
    src_idx += (remaining % dim) * s;
    remaining = remaining / dim;
  }

  dst[flat] = src[src_idx];
}
