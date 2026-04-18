import source from './compare.wgsl';
import { compareKernel } from './_factory';

export const eqKernel = compareKernel('Equal', source, 'av == bv');
