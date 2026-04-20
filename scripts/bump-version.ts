// Bumps the version of every publishable @webtensor/* package in lockstep.
//
// Usage:
//   bun scripts/bump-version.ts patch       # 0.3.3 -> 0.3.4
//   bun scripts/bump-version.ts minor       # 0.3.3 -> 0.4.0
//   bun scripts/bump-version.ts major       # 0.3.3 -> 1.0.0
//   bun scripts/bump-version.ts 0.4.2       # explicit version
//
// All seven packages share the same version — workspace deps use `workspace:^`
// so inter-package refs resolve automatically at publish time.

import { readFileSync, writeFileSync } from 'node:fs';

const PACKAGES = [
  'packages/ir',
  'packages/runtime',
  'packages/core',
  'packages/backend-cpu',
  'packages/backend-webgpu',
  'packages/backend-wasm',
  'packages/nn',
];

const arg = process.argv[2];
if (!arg) {
  console.error('Usage: bun scripts/bump-version.ts <patch|minor|major|x.y.z>');
  process.exit(1);
}

const current = JSON.parse(readFileSync(`${PACKAGES[0]}/package.json`, 'utf8')).version as string;

for (const p of PACKAGES) {
  const v = JSON.parse(readFileSync(`${p}/package.json`, 'utf8')).version;
  if (v !== current) {
    console.error(`version drift: ${p} is ${v}, expected ${current}`);
    process.exit(1);
  }
}

function bump(v: string, kind: string): string {
  if (/^\d+\.\d+\.\d+$/.test(kind)) return kind;
  const [maj, min, pat] = v.split('.').map(Number);
  if (kind === 'patch') return `${maj}.${min}.${pat + 1}`;
  if (kind === 'minor') return `${maj}.${min + 1}.0`;
  if (kind === 'major') return `${maj + 1}.0.0`;
  throw new Error(`unknown bump kind: ${kind}`);
}

const next = bump(current, arg);
console.log(`${current} -> ${next}`);

for (const p of PACKAGES) {
  const path = `${p}/package.json`;
  const src = readFileSync(path, 'utf8');
  const pkg = JSON.parse(src);
  pkg.version = next;
  writeFileSync(path, JSON.stringify(pkg, null, 2) + '\n');
  console.log(`  ${p} -> ${next}`);
}
