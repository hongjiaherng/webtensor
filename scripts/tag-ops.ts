// One-shot helper: add `@category <Group>` to the JSDoc of every exported
// function in packages/core/src/{ops,init,autograd,compile,equality}.
// Idempotent — skips files that already have an @category tag.
//
// Run: bun scripts/tag-ops.ts

export {};

import { Glob } from 'bun';

const CATEGORIES: Array<[RegExp, string]> = [
  [/packages\/core\/src\/ops\/elementwise\//, 'Elementwise'],
  [/packages\/core\/src\/ops\/reduction\//, 'Reduction'],
  [/packages\/core\/src\/ops\/linalg\//, 'Linalg'],
  [/packages\/core\/src\/ops\/activation\//, 'Activation'],
  [/packages\/core\/src\/ops\/movement\//, 'Movement'],
  [/packages\/core\/src\/ops\/memory\//, 'Memory'],
  [/packages\/core\/src\/init\//, 'Factories'],
  [/packages\/core\/src\/autograd\//, 'Autograd'],
  [/packages\/core\/src\/compile\//, 'Compile'],
  [/packages\/core\/src\/equality\.ts$/, 'Autograd'],
  [/packages\/core\/src\/run\.ts$/, 'Compile'],
  [/packages\/core\/src\/tensor\.ts$/, 'Tensor'],
];

const BLOCK_RE =
  /(\/\*\*[\s\S]*?\*\/\s*\n\s*)?export\s+(function|class|const)\s+([A-Za-z_][\w]*)/g;

let changed = 0;
let skipped = 0;

for await (const file of new Glob('packages/core/src/**/*.ts').scan('.')) {
  if (file.endsWith('/index.ts') || /\/_[^/]+\.ts$/.test(file)) continue;
  const category = CATEGORIES.find(([re]) => re.test(file.replace(/\\/g, '/')))?.[1];
  if (!category) continue;

  const src = await Bun.file(file).text();
  if (src.includes('@category')) {
    skipped++;
    continue;
  }

  const next = src.replace(BLOCK_RE, (match, doc) => {
    if (doc) {
      // Multi-line JSDoc: insert before closing */
      if (/\n\s*\*\//.test(doc)) {
        return match.replace(
          /(\n\s*)\*\/(\s*\n\s*export)/,
          (_m, ws, tail) => `${ws}* @category ${category}${ws}*/${tail}`,
        );
      }
      // Single-line JSDoc (/** ... */ on one line): expand it.
      return match.replace(
        /\/\*\*\s*(.*?)\s*\*\/\s*\n/,
        (_m, body) => `/**\n * ${body}\n * @category ${category}\n */\n`,
      );
    }
    return `/**\n * @category ${category}\n */\n${match}`;
  });

  if (next !== src) {
    await Bun.write(file, next);
    console.log(`  tagged ${file}`);
    changed++;
  }
}

console.log(`\n✓ tagged ${changed} files (${skipped} already tagged)`);
