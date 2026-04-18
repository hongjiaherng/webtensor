export * from './backend';

import { registerBackend } from '@webtensor/runtime';
import { CPUBackend } from './backend';

registerBackend('cpu', () => CPUBackend.create());
