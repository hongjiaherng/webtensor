import source from './compare.wgsl';
import { compareKernel } from './_factory';

export const geKernel = compareKernel('GreaterOrEqual', source, 'av >= bv');
