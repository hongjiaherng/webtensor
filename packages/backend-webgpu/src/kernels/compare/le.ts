import source from './compare.wgsl';
import { compareKernel } from './_factory';

export const leKernel = compareKernel('LessOrEqual', source, 'av <= bv');
