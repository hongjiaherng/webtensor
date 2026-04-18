import source from './add.wgsl';
import { binaryKernel } from './_helpers';

export const addKernel = binaryKernel('Add', source);
