// @ts-check
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import prettierConfig from 'eslint-config-prettier';
import unusedImports from 'eslint-plugin-unused-imports';

export default tseslint.config(
  js.configs.recommended,
  tseslint.configs.recommended,

  // Project-wide settings
  {
    languageOptions: {
      parserOptions: {
        projectService: {
          // Files not covered by any tsconfig — parsed with default compiler options
          // instead of erroring out. Includes loose .ts files in scripts/ and per-package
          // build.ts files (e.g. packages/backend-wasm/build.ts).
          allowDefaultProject: ['*.js', '*.ts', 'scripts/*.ts', 'packages/*/build.ts'],
        },
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },

  // Unused imports — auto-fixable with --fix; disables the duplicate core rules
  {
    plugins: { 'unused-imports': unusedImports },
    rules: {
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
      'unused-imports/no-unused-imports': 'error',
      'unused-imports/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
    },
  },

  // Relaxations that make sense for a library / research project
  {
    rules: {
      // `any` is sometimes unavoidable when bridging WASM/WebGPU boundaries
      '@typescript-eslint/no-explicit-any': 'warn',
      // Non-null assertions are intentional in hot-path kernel code
      '@typescript-eslint/no-non-null-assertion': 'off',
    },
  },

  // Disable formatting rules that conflict with Prettier — must be last
  prettierConfig,

  // Ignore generated, compiled, third-party, and self-contained subprojects
  // (docs/ and smoke-test/ have their own eslint setups and are linted via their own scripts)
  {
    ignores: [
      '**/dist/',
      'node_modules/',
      'packages/backend-wasm/pkg/',
      'packages/backend-wasm/rust/',
      'docs/',
      'smoke-test/',
      'bun.lock',
      '**/*.wgsl',
    ],
  },
);
