import source from './add.wgsl';
import { binaryKernel } from './_factory';

export const addKernel = binaryKernel('Add', source);
