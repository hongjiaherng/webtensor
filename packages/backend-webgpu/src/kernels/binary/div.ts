import source from './div.wgsl';
import { binaryKernel } from './_factory';

export const divKernel = binaryKernel('Div', source);
