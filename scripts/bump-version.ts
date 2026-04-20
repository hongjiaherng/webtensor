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
export {}; // mark as module so top-level `await` is legal

// Publishable packages bumped in strict lockstep — they share a version and
// reference each other via `workspace:^`, so drift would break publish.
const PACKAGES = [
  'packages/ir',
  'packages/runtime',
  'packages/core',
  'packages/backend-cpu',
  'packages/backend-webgpu',
  'packages/backend-wasm',
  'packages/nn',
];

// Private packages we keep aligned for consistency but don't enforce drift on,
// since they've historically had independent versions. They get the same
// `next` version written without comparing their current state.
const MIRROR_PACKAGES = ['.', 'docs', 'smoke-test'];

const arg = Bun.argv[2];
if (!arg) {
  console.error('Usage: bun scripts/bump-version.ts <patch|minor|major|x.y.z>');
  process.exit(1);
}

async function readPkg(dir: string): Promise<{ version: string } & Record<string, unknown>> {
  return await Bun.file(`${dir}/package.json`).json();
}

const current = (await readPkg(PACKAGES[0])).version;

for (const p of PACKAGES) {
  const v = (await readPkg(p)).version;
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
  const pkg = await readPkg(p);
  const prev = pkg.version;
  pkg.version = next;
  await Bun.write(path, JSON.stringify(pkg, null, 2) + '\n');
  console.log(`  ${p} -> ${next} ${prev !== current ? `(was ${current})` : ''}`);
}

for (const p of MIRROR_PACKAGES) {
  const path = `${p}/package.json`;
  const pkg = await readPkg(p);
  const prev = pkg.version;
  pkg.version = next;
  await Bun.write(path, JSON.stringify(pkg, null, 2) + '\n');
  console.log(
    `  ${p === '.' ? '(root)' : p} -> ${next}${prev !== current ? ` (was ${prev})` : ''}`,
  );
}
