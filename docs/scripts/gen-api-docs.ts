// Generates API reference markdown from TypeDoc into content/docs/api/,
// then post-processes the output to fit fumadocs:
//   1. flatten the trailing `src/` directory each package gets
//   2. add minimal frontmatter (title, description) to every .mdx
//   3. reorganize files into @category folders (elementwise/, reduction/, ...)
//      based on the grouping TypeDoc emits in each package README
//   4. rewrite all internal markdown links to match new layout, strip `.mdx`
//      and stray `/src/` segments so fumadocs routing resolves
//   5. emit meta.json per folder with title + description; category/package
//      landing pages are frontmatter-only — the React <FolderOverview>
//      component renders child cards from the page tree at request time.
//
// Run from the `docs/` workspace: `bun scripts/gen-api-docs.ts`
// or indirectly via `bun run docs:api`.

export {}; // module marker for top-level await

import { $, Glob } from 'bun';
import { readdir, rename, rm, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const API_ROOT = 'content/docs/api';
// URL prefix under which API pages are served (matches `docsRoute` + `/api`).
const API_URL = '/docs/api';

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
const H1 = /^#\s+(.+?)\s*$/m;

for await (const file of new Glob(`${API_ROOT}/**/*.mdx`).scan('.')) {
  const raw = await Bun.file(file).text();
  if (raw.startsWith('---')) continue; // already has frontmatter
  const m = raw.match(H1);

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
  // TypeDoc conservatively escapes `<`/`>`/`_` everywhere (e.g.
  // `NestedArray\<T\>`, `MAX\_RANK`). `<>` only need escaping inside MDX
  // prose — unescape them inside fenced/inline code. `_` is safe to
  // unescape everywhere: intraword underscores aren't italic markers in
  // CommonMark, and identifiers never look like emphasis delimiters.
  const cleanTitle = title.replace(/\\([<>_])/g, '$1');
  const cleanBody = body
    .replace(/\\_/g, '_')
    .replace(/```[\s\S]*?```|`[^`\n]*`/g, (block) => block.replace(/\\([<>])/g, '$1'));
  const fm = ['---', `title: '${cleanTitle.replace(/'/g, "\\'")}'`, '---', ''].join('\n');
  await Bun.write(file, fm + cleanBody);
}

// ---------------------------------------------------------------------------
// 3. Build the move map by parsing each package's README.
//    TypeDoc emits grouped bullet lists per @category — we use those groups to
//    decide the new physical folder for each page. Entries without a category
//    (or under "Other") keep their member-kind folder.

type Moves = Map<string, string>; // api-root-relative, no .mdx extension

const kebab = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');

function parseCategoryMap(readme: string): Map<string, string> {
  const map = new Map<string, string>();
  let currentCat: string | null = null;
  for (const rawLine of readme.split('\n')) {
    const h2 = rawLine.match(/^##\s+/);
    if (h2) {
      currentCat = null;
      continue;
    }
    const h3 = rawLine.match(/^###\s+(.+?)\s*$/);
    if (h3) {
      currentCat = h3[1];
      continue;
    }
    if (!currentCat) continue;
    // - [name](...) or - [~~name~~](...)  (deprecated)
    const bullet = rawLine.match(/^- \[~?~?([^~\]]+?)~?~?\]\(/);
    if (bullet) map.set(bullet[1], currentCat);
  }
  return map;
}

const moves: Moves = new Map();

// Reserved category slugs we prefer over member-kind folders.
const KNOWN_CATS = new Set([
  'tensor',
  'factories',
  'elementwise',
  'reduction',
  'linalg',
  'activation',
  'movement',
  'memory',
  'autograd',
  'compile',
  'runtime',
]);

for (const pkg of pkgs) {
  const readmePath = path.join(API_ROOT, pkg, 'README.mdx');
  if (!existsSync(readmePath)) continue;
  const catMap = parseCategoryMap(await Bun.file(readmePath).text());

  for await (const file of new Glob(`${pkg}/**/*.mdx`).scan(API_ROOT)) {
    if (file.endsWith('README.mdx')) continue;
    const posixFile = file.replaceAll('\\', '/');
    const parts = posixFile.split('/');
    if (parts.length !== 3) continue; // expect <pkg>/<kind>/<name>.mdx
    const [, kind, base] = parts;
    const name = base.replace(/\.mdx$/, '');
    const cat = catMap.get(name);
    const slug = cat ? kebab(cat) : '';
    const newKind = slug && KNOWN_CATS.has(slug) ? slug : kind;
    const oldAbs = `${pkg}/${kind}/${name}`;
    const newAbs = `${pkg}/${newKind}/${name}`;
    if (oldAbs !== newAbs) moves.set(oldAbs, newAbs);
  }
}

// ---------------------------------------------------------------------------
// 4. Rewrite all internal links, then physically move files.

function rewriteLinks(content: string, sourceAbs: string): string {
  // TypeDoc generates relative links calibrated for the ORIGINAL file layout
  // (with a `src/` segment between <pkg>/ and the member-kind folder). We've
  // since flattened that `src/` away, so joining those links against the new
  // sourceDir miscounts the depth. Reconstruct the pre-flatten sourceDir by
  // re-inserting `src/` after the package segment, resolve the link there,
  // then strip `/src/` to land on the post-flatten abs path.
  const preFlattenSourceDir = path.posix
    .dirname(sourceAbs)
    .replace(/^([^/]+)(\/|$)/, '$1/src$2');
  return content.replace(/\]\(([^)]+)\)/g, (match, link) => {
    if (/^(https?:|mailto:|#|\/)/.test(link)) return match;
    const [pathPart, ...rest] = link.split('#');
    const hash = rest.length ? '#' + rest.join('#') : '';
    if (!pathPart) return match;

    let abs = path.posix.normalize(path.posix.join(preFlattenSourceDir, pathPart));
    abs = abs.replace(/\.mdx$/, '');
    // Strip any `/src/` segment left over from cross-package refs.
    abs = abs.replace(/\/src\//g, '/').replace(/^src\//, '');

    const moved = moves.get(abs);
    if (moved) abs = moved;

    // Use absolute URLs — relative hrefs resolve against the browser's current
    // path, which strips the trailing segment and miscomputes cross-package
    // links. Absolute paths always resolve correctly.
    return `](${API_URL}/${abs}${hash})`;
  });
}

for await (const file of new Glob(`${API_ROOT}/**/*.mdx`).scan('.')) {
  const posixFile = file.replaceAll('\\', '/');
  const sourceAbs = posixFile
    .replace(/^.*\/api\//, '')
    .replace(/\.mdx$/, '');
  const content = await Bun.file(file).text();
  const rewritten = rewriteLinks(content, sourceAbs);

  const newAbs = moves.get(sourceAbs);
  if (newAbs) {
    const destRel = path.join(API_ROOT, ...newAbs.split('/')) + '.mdx';
    await mkdir(path.dirname(destRel), { recursive: true });
    await Bun.write(destRel, rewritten);
    if (path.resolve(file) !== path.resolve(destRel)) {
      await rm(file, { force: true });
    }
  } else if (rewritten !== content) {
    await Bun.write(file, rewritten);
  }
}

// Clean up now-empty member-kind folders.
for (const pkg of pkgs) {
  const pkgDir = path.join(API_ROOT, pkg);
  for (const kind of ['functions', 'classes', 'interfaces', 'type-aliases', 'variables']) {
    const kindDir = path.join(pkgDir, kind);
    if (!existsSync(kindDir)) continue;
    const entries = await readdir(kindDir);
    if (entries.length === 0) await rm(kindDir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// 5. Landing pages are frontmatter-only stubs. The FolderOverview React
//    component reads the page tree and renders child cards at request time,
//    pulling descriptions from each folder/page's meta.json / frontmatter.
//    No JSX is injected into generated MDX.

// Human-readable title + blurb per category folder. Blurbs flow into each
// category folder's meta.json `description`, which the page tree exposes so
// FolderOverview can show them on the package landing page.
const FOLDER_META: Record<string, { title: string; description: string }> = {
  tensor: { title: 'Tensor', description: 'The Tensor class and its methods.' },
  factories: {
    title: 'Factories',
    description: 'Create tensors from data, shapes, or distributions.',
  },
  elementwise: {
    title: 'Elementwise',
    description: 'Per-element math, comparisons, and casting with broadcasting.',
  },
  reduction: {
    title: 'Reduction',
    description: 'Collapse axes: sum, mean, all, any.',
  },
  linalg: { title: 'Linalg', description: 'Matrix multiplication and linear algebra.' },
  activation: {
    title: 'Activation',
    description: 'Nonlinearities — relu, sigmoid, tanh, softmax.',
  },
  movement: {
    title: 'Movement',
    description: 'Reshape, transpose, slice, concat, and other zero-copy view ops.',
  },
  memory: {
    title: 'Memory',
    description: 'Copy, clone, and detach — control tensor identity and backing storage.',
  },
  autograd: { title: 'Autograd', description: 'Gradients and reverse-mode differentiation.' },
  compile: {
    title: 'Compile',
    description: 'Trace a function into a reusable graph with compile() and run().',
  },
  runtime: { title: 'Runtime', description: 'Engine, backends, and device execution.' },
  classes: { title: 'Classes', description: 'Exported classes.' },
  interfaces: { title: 'Interfaces', description: 'Exported interfaces and option bags.' },
  'type-aliases': { title: 'Type Aliases', description: 'Exported type aliases.' },
  variables: { title: 'Variables', description: 'Exported constants.' },
  functions: { title: 'Functions', description: 'Uncategorized functions.' },
};

// Short tagline per package. Flows into each package folder's meta.json
// `description`, surfaced by FolderOverview on the API root landing page.
const PACKAGE_META: Record<string, string> = {
  core: 'User-facing Tensor class, eager ops, autograd, compile().',
  runtime: 'Engine, Backend interface, tensor lifecycle, view ops.',
  ir: 'Pure data: Node, Value, Graph. No devices, no gradients.',
  'backend-cpu': 'TypeScript kernels.',
  'backend-wasm': 'Rust kernels compiled to WebAssembly.',
  'backend-webgpu': 'WGSL compute shaders.',
  nn: 'Losses and optimizers (mseLoss, SGD).',
};

async function writeStubIndex(dir: string, title: string): Promise<void> {
  const fm = ['---', `title: '${title.replace(/'/g, "\\'")}'`, '---', ''].join('\n');
  await rm(path.join(dir, 'README.mdx'), { force: true });
  await Bun.write(path.join(dir, 'index.mdx'), fm);
}

// Clear any stale per-folder index.mdx from previous script runs.
for await (const stale of new Glob(`${API_ROOT}/**/index.mdx`).scan('.')) {
  await rm(stale, { force: true });
}

for (const pkg of pkgs) {
  await writeStubIndex(path.join(API_ROOT, pkg), `@webtensor/${pkg}`);
}
await writeStubIndex(API_ROOT, 'API Reference');

// ---------------------------------------------------------------------------
// 6. Emit a meta.json per folder. Categories within a package are grouped
//    using fumadocs' `---Title---` separator syntax; folders carry their
//    description so the page tree exposes it for FolderOverview's cards.

const SIDEBAR_SECTIONS: { title: string; entries: string[] }[] = [
  { title: 'Core', entries: ['tensor', 'factories'] },
  {
    title: 'Ops',
    entries: ['elementwise', 'reduction', 'linalg', 'activation', 'movement', 'memory'],
  },
  { title: 'Execution', entries: ['autograd', 'compile', 'runtime'] },
  {
    title: 'Types',
    entries: ['classes', 'interfaces', 'type-aliases', 'variables', 'functions'],
  },
];

function groupedPackagePages(entries: string[]): string[] {
  const leftover = new Set(entries);
  const pages: string[] = [];
  // Put the folder landing page first (hidden from sidebar via fumadocs index merge).
  if (leftover.has('index')) {
    pages.push('index');
    leftover.delete('index');
  }
  for (const section of SIDEBAR_SECTIONS) {
    const present = section.entries.filter((e) => leftover.has(e));
    if (present.length === 0) continue;
    pages.push(`---${section.title}---`);
    for (const e of present) {
      pages.push(e);
      leftover.delete(e);
    }
  }
  if (leftover.size > 0) {
    pages.push('---Other---');
    for (const e of [...leftover].sort()) pages.push(e);
  }
  return pages;
}

async function writeMeta(
  dir: string,
  title: string,
  opts: { grouped?: boolean; root?: boolean; description?: string } = {},
): Promise<void> {
  const entries = (await readdir(dir, { withFileTypes: true }))
    .map((d) => (d.isDirectory() ? d.name : d.name.replace(/\.mdx$/, '')))
    // Exclude the folder-level landing page itself — fumadocs merges it with
    // the folder node so it shouldn't appear as a sibling sidebar entry.
    .filter((n) => n !== 'meta' && n !== 'README');
  const sorted = entries.filter((n) => n !== 'index').sort();
  const pages = opts.grouped ? groupedPackagePages(entries) : (entries.includes('index') ? ['index', ...sorted] : sorted);
  const meta: Record<string, unknown> = { title, pages };
  if (opts.description) meta.description = opts.description;
  if (opts.root) meta.root = true;
  await Bun.write(path.join(dir, 'meta.json'), JSON.stringify(meta, null, 2) + '\n');
}

for (const pkg of pkgs) {
  const pkgDir = path.join(API_ROOT, pkg);
  const sections = (await readdir(pkgDir, { withFileTypes: true })).filter((d) => d.isDirectory());
  for (const s of sections) {
    const meta = FOLDER_META[s.name];
    const title = meta?.title ?? s.name.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    await writeMeta(path.join(pkgDir, s.name), title, { description: meta?.description });
  }
  await writeMeta(pkgDir, `@webtensor/${pkg}`, {
    grouped: true,
    description: PACKAGE_META[pkg],
  });
}

// `root: true` promotes this folder to its own sidebar tab (DocsLayout tabs
// dropdown) — keeps the API reference visually separate from curated docs.
await writeMeta(API_ROOT, 'API Reference', { root: true });

console.log('✓ api docs generated at', API_ROOT);
