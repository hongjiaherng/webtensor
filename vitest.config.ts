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
      '@minitensor/runtime': resolve(__dirname, 'packages/runtime/src/index.ts'),
      '@minitensor/ir': resolve(__dirname, 'packages/ir/src/index.ts'),
      '@minitensor/core': resolve(__dirname, 'packages/core/src/index.ts'),
      '@minitensor/backend-cpu': resolve(__dirname, 'packages/backend-cpu/src/index.ts'),
      '@minitensor/backend-wasm': resolve(__dirname, 'packages/backend-wasm/src/index.ts'),
      '@minitensor/backend-webgpu': resolve(__dirname, 'packages/backend-webgpu/src/index.ts'),
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
