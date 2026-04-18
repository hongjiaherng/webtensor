import source from './_compare.wgsl';
import { compareKernel } from './_helpers';

export const neKernel = compareKernel('NotEqual', source, 'av != bv');
