import source from './sub.wgsl';
import { binaryKernel } from './_factory';

export const subKernel = binaryKernel('Sub', source);
