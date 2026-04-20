// Generates API reference markdown from TypeDoc into content/docs/api/,
// then post-processes the output to fit fumadocs:
//   1. flatten the trailing `src/` directory each package gets
//   2. add minimal frontmatter (title, description) to every .mdx
//   3. generate meta.json per folder so the sidebar is stable
//
// Run from the `docs/` workspace: `bun scripts/gen-api-docs.ts`
// or indirectly via `bun run docs:api`.

export {}; // module marker for top-level await

import { $, Glob } from 'bun';
import { readdir, rename, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const API_ROOT = 'content/docs/api';

console.log('→ running typedoc');
await $`bun x typedoc`;

// ---------------------------------------------------------------------------
// 1. Flatten the trailing `src/` each package entry creates.
const pkgs = (await readdir(API_ROOT, { withFileTypes: true }))
  .filter((d) => d.isDirectory())
  .map((d) => d.name);

for (const pkg of pkgs) {
  const srcDir = path.join(API_ROOT, pkg, 'src');
  if (!existsSync(srcDir)) continue;
  for (const entry of await readdir(srcDir)) {
    await rename(path.join(srcDir, entry), path.join(API_ROOT, pkg, entry));
  }
  await rm(srcDir, { recursive: true, force: true });
}

// ---------------------------------------------------------------------------
// 2. Add frontmatter to every generated .mdx. TypeDoc emits an H1 as the
//    first line (e.g. `# Function: add()`) — promote that to frontmatter.title
//    and drop the H1 so fumadocs renders its own page header.
const mdxGlob = new Glob(`${API_ROOT}/**/*.mdx`);
const H1 = /^#\s+(.+?)\s*$/m;

for await (const file of mdxGlob.scan('.')) {
  const raw = await Bun.file(file).text();
  if (raw.startsWith('---')) continue; // already has frontmatter
  const m = raw.match(H1);

  // README.mdx files land at the package root with titles like `core/src`.
  // Retitle them to `@webtensor/<pkg>` for a cleaner TOC page.
  const isPkgReadme = /[\\/]api[\\/]([^\\/]+)[\\/]README\.mdx$/.test(file);
  const pkgMatch = file.match(/[\\/]api[\\/]([^\\/]+)[\\/]README\.mdx$/);

  let title: string;
  if (isPkgReadme && pkgMatch) {
    title = `@webtensor/${pkgMatch[1]}`;
  } else {
    title = m
      ? m[1]
          .replace(/^(Function|Class|Interface|Type Alias|Variable|Enumeration):\s*/, '')
          .replace(/\(\)$/, '')
      : path.basename(file, '.mdx');
  }

  const body = m ? raw.replace(m[0], '').replace(/^\n+/, '') : raw;
  const fm = ['---', `title: '${title.replace(/'/g, "\\'")}'`, '---', ''].join('\n');
  await writeFile(file, fm + body);
}

// ---------------------------------------------------------------------------
// 3. Emit a meta.json per folder so the sidebar has deterministic order.
async function writeMeta(dir: string, title: string): Promise<void> {
  const entries = (await readdir(dir, { withFileTypes: true }))
    .map((d) => (d.isDirectory() ? d.name : d.name.replace(/\.mdx$/, '')))
    .filter((n) => n !== 'meta')
    .sort();
  const pages = entries.includes('README') ? ['README', ...entries.filter((e) => e !== 'README')] : entries;
  await writeFile(path.join(dir, 'meta.json'), JSON.stringify({ title, pages }, null, 2) + '\n');
}

for (const pkg of pkgs) {
  const pkgDir = path.join(API_ROOT, pkg);
  const sections = (await readdir(pkgDir, { withFileTypes: true })).filter((d) => d.isDirectory());
  for (const s of sections) {
    const title = s.name.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    await writeMeta(path.join(pkgDir, s.name), title);
  }
  await writeMeta(pkgDir, `@webtensor/${pkg}`);
}

await writeMeta(API_ROOT, 'API Reference');

console.log('✓ api docs generated at', API_ROOT);
