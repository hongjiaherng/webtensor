import source from './mul.wgsl';
import { binaryKernel } from './_helpers';

export const mulKernel = binaryKernel('Mul', source);
