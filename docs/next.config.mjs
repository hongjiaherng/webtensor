import { createMDX } from 'fumadocs-mdx/next';
import path from 'path';
import { fileURLToPath } from 'url';

const withMDX = createMDX();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const basePath = process.env.DEPLOY_BASE_PATH ?? '';

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  basePath,
  env: {
    NEXT_PUBLIC_BASE_PATH: basePath,
  },
  turbopack: {
    root: path.join(__dirname, '..'),
  },
  reactStrictMode: true,
};

export default withMDX(config);
