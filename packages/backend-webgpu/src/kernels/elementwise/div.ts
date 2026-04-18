import source from './div.wgsl';
import { binaryKernel } from './_helpers';

export const divKernel = binaryKernel('Div', source);
