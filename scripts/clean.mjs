import { rmSync } from 'fs';

const all = process.argv.includes('--all');

// Build artifacts
const dirs = [
  'packages/ir/dist',
  'packages/runtime/dist',
  'packages/core/dist',
  'packages/backend-cpu/dist',
  'packages/backend-webgpu/dist',
  'packages/backend-wasm/dist',
  'packages/backend-wasm/pkg',
];

if (all) {
  dirs.push(
    'packages/backend-wasm/rust/target',
    'packages/backend-wasm/rust/Cargo.lock',
    'node_modules',
    'bun.lock',
  );
}

for (const d of dirs) {
  rmSync(d, { recursive: true, force: true });
}
