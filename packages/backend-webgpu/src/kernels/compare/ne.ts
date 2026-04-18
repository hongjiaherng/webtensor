import source from './compare.wgsl';
import { compareKernel } from './_factory';

export const neKernel = compareKernel('NotEqual', source, 'av != bv');
