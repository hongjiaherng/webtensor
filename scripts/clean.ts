import { $ } from 'bun';

const all = process.argv.includes('--all');

const dirs: string[] = [
  'packages/ir/dist',
  'packages/runtime/dist',
  'packages/core/dist',
  'packages/backend-cpu/dist',
  'packages/backend-webgpu/dist',
  'packages/backend-wasm/dist',
  'packages/backend-wasm/pkg',
  'packages/nn/dist',
  'docs/out',
  'docs/.next',
  'docs/.source',
];

if (all) {
  dirs.push(
    'packages/backend-wasm/rust/target',
    'assets/logo/.venv',
    // Don't delete lock file to avoid breaking reproducible builds, but it can be deleted if you want to regenerate it.
    // 'packages/backend-wasm/rust/Cargo.lock',
    // 'bun.lock',
  );
}

for (const d of dirs) {
  console.log(`removing ${d}`);
  await $`rm -rf ${d}`.quiet().nothrow();
}

if (all) {
  // Remind user to delete node_modules manually (node_modules, docs/node_modules)
  console.log(
    'Please delete node_modules and docs/node_modules manually if you want to clean them as well.',
  );
}
