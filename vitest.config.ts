import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
import wasm from 'vite-plugin-wasm';
import { resolve } from 'path';

export default defineConfig({
  plugins: [wasm()],
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
