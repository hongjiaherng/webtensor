import { defineConfig, Plugin } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
import { resolve } from 'path';
import { readFileSync } from 'fs';

/** Load .wgsl files as text (mirrors bun build --loader:.wgsl=text) */
function wgslTextPlugin(): Plugin {
  return {
    name: 'wgsl-text',
    transform(_code, id) {
      if (id.endsWith('.wgsl')) {
        const src = readFileSync(id, 'utf8');
        return { code: `export default ${JSON.stringify(src)};`, map: null };
      }
    },
  };
}

/** Inline .wasm as a base64 string default export — mirrors backend-wasm/build.ts */
function wasmInlineBase64Plugin(): Plugin {
  return {
    name: 'wasm-inline-base64',
    enforce: 'pre',
    load(id) {
      const path = id.split('?')[0];
      if (path.endsWith('.wasm')) {
        const base64 = readFileSync(path).toString('base64');
        return `export default ${JSON.stringify(base64)};`;
      }
    },
  };
}

export default defineConfig({
  plugins: [wasmInlineBase64Plugin(), wgslTextPlugin()],
  resolve: {
    alias: {
      // Resolve all workspace packages directly to TypeScript source so that
      // Vite never needs a pre-built dist/ directory.  This makes runtime
      // value imports (functions, not just types) work correctly.
      '@webtensor/runtime': resolve(__dirname, 'packages/runtime/src/index.ts'),
      '@webtensor/ir': resolve(__dirname, 'packages/ir/src/index.ts'),
      '@webtensor/core': resolve(__dirname, 'packages/core/src/index.ts'),
      '@webtensor/backend-cpu': resolve(__dirname, 'packages/backend-cpu/src/index.ts'),
      '@webtensor/backend-wasm': resolve(__dirname, 'packages/backend-wasm/src/index.ts'),
      '@webtensor/backend-webgpu': resolve(__dirname, 'packages/backend-webgpu/src/index.ts'),
    },
  },
  test: {
    browser: {
      enabled: true,
      provider: playwright({
        launchOptions: {
          args: ['--enable-unsafe-webgpu'],
        },
      }),
      instances: [{ browser: 'chromium' }],
      headless: false,
    },
    include: ['tests/**/*.test.ts'],
  },
});
