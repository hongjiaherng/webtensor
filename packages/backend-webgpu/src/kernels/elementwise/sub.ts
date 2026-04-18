import source from './sub.wgsl';
import { binaryKernel } from './_helpers';

export const subKernel = binaryKernel('Sub', source);
