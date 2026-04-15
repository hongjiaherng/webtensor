import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';

export default defineConfig({
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
