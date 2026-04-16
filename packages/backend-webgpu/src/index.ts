export * from './backend';

import { registerBackend } from '@webtensor/runtime';
import { WebGPUBackend } from './backend';

registerBackend('webgpu', () => WebGPUBackend.create());
