import source from './_compare.wgsl';
import { compareKernel } from './_helpers';

export const gtKernel = compareKernel('Greater', source, 'av > bv');
