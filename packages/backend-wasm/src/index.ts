export * from './backend';

import { registerBackend } from '@webtensor/runtime';
import { WASMBackend } from './backend';

registerBackend('wasm', () => WASMBackend.create());
