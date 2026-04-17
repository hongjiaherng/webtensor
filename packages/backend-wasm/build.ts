import { readFileSync } from 'fs';

const result = await Bun.build({
  entrypoints: ['./src/index.ts'],
  outdir: './dist',
  target: 'browser',
  format: 'esm',
  external: ['@webtensor/ir', '@webtensor/runtime'],
  plugins: [
    {
      name: 'wasm-inline',
      setup(build) {
        build.onLoad({ filter: /\.wasm$/ }, (args) => {
          const base64 = readFileSync(args.path).toString('base64');
          return {
            contents: `export default ${JSON.stringify(base64)};`,
            loader: 'js',
          };
        });
      },
    },
  ],
});

if (!result.success) {
  console.error(result.logs);
  process.exit(1);
}
