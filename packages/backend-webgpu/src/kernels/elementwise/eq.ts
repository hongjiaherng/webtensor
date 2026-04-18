import source from './_compare.wgsl';
import { compareKernel } from './_helpers';

export const eqKernel = compareKernel('Equal', source, 'av == bv');
