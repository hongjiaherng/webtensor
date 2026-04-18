import source from './_compare.wgsl';
import { compareKernel } from './_helpers';

export const leKernel = compareKernel('LessOrEqual', source, 'av <= bv');
