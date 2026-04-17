import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  basePath: process.env.NODE_ENV === 'production' ? '/webtensor' : '',
  reactStrictMode: true,
};

export default withMDX(config);
