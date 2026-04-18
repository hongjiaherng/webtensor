import source from './mul.wgsl';
import { binaryKernel } from './_factory';

export const mulKernel = binaryKernel('Mul', source);
