import source from './_compare.wgsl';
import { compareKernel } from './_helpers';

export const ltKernel = compareKernel('Less', source, 'av < bv');
