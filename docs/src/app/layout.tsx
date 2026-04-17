import { Inter } from 'next/font/google';
import { Provider } from '@/components/provider';
import './global.css';

const inter = Inter({
  subsets: ['latin'],
});

const deployBasePath = process.env.DEPLOY_BASE_PATH ?? '';
const baseUrl = deployBasePath
  ? `https://hongjiaherng.github.io${deployBasePath}`
  : 'http://localhost:3000';

const faviconPath = `${deployBasePath}/favicon.svg`;

export const metadata = {
  title: 'webtensor',
  description: 'A tensor library that runs entirely in the browser',
  metadataBase: new URL(baseUrl),
  icons: {
    icon: [{ url: faviconPath, type: 'image/svg+xml' }],
    apple: faviconPath,
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
