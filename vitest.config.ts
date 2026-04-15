import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
  plugins: [wasm()],
  test: {
    browser: {
      enabled: true,
      provider: playwright({
        launchOptions: {
          args: ['--enable-unsafe-webgpu']
        }
      }),
      instances: [
        { browser: 'chromium' },
      ],
      headless: false,
    },
    include: ['tests/**/*.test.ts'],
  },
});
