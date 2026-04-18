import source from './compare.wgsl';
import { compareKernel } from './_factory';

export const ltKernel = compareKernel('Less', source, 'av < bv');
