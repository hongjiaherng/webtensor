/// Compute the flat storage index for the `flat_idx`-th output element
/// given a shape, broadcast strides, and a base offset.
///
/// Iterates from innermost to outermost axis, decomposing `flat_idx` into
/// per-axis coordinates and dotting with `strides`. Broadcast axes have
/// stride 0, so repeated reads return the same element.
pub fn strided_idx(shape: &[u32], strides: &[u32], offset: u32, flat_idx: u32) -> usize {
    let rank = shape.len();
    let mut rem = flat_idx;
    let mut idx = offset as usize;
    for d in 0..rank {
        let axis = rank - 1 - d;
        let coord = rem % shape[axis];
        rem /= shape[axis];
        idx += (coord * strides[axis]) as usize;
    }
    idx
}
