import source from './_compare.wgsl';
import { compareKernel } from './_helpers';

export const geKernel = compareKernel('GreaterOrEqual', source, 'av >= bv');
