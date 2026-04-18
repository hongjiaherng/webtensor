import {
  WASMKernel,
  handleOf,
  allocMeta,
  buildReduceMetaData,
  dtypeSuffix,
} from '../utils';
import type { WebtensorWasmModule } from '../../module';

export const reduceSumKernel: WASMKernel = (module, node, inputs, outputs) => {
  const axes = (node.attributes?.axes as number[]) ?? [];
  const meta = buildReduceMetaData(inputs, axes);
  const metaPtr = allocMeta(module, meta);
  try {
    const a = handleOf(inputs[0]);
    const out = handleOf(outputs[0]);
    const suffix = dtypeSuffix(inputs[0].dtype);
    const fn = (module as unknown as Record<string, WebtensorWasmModule['reduce_sum_f32_strided']>)[
      `reduce_sum_${suffix}_strided`
    ];
    if (!fn) {
      throw new Error(`reduceSum: no WASM kernel for dtype ${inputs[0].dtype}`);
    }
    fn(a.ptr, out.ptr, metaPtr);
  } finally {
    module.free_u32(metaPtr, meta.length);
  }
};
