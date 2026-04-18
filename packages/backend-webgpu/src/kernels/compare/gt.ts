import source from './compare.wgsl';
import { compareKernel } from './_factory';

export const gtKernel = compareKernel('Greater', source, 'av > bv');
