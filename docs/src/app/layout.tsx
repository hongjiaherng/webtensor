import { Inter } from 'next/font/google';
import { Provider } from '@/components/provider';
import './global.css';

const inter = Inter({
  subsets: ['latin'],
});

const baseUrl = process.env.NODE_ENV === 'production'
  ? 'https://hongjiaherng.github.io/webtensor'
  : 'http://localhost:3000';

export const metadata = {
  title: 'webtensor',
  description: 'A tensor library that runs entirely in the browser',
  metadataBase: new URL(baseUrl),
  icons: {
    icon: [
      { url: '/favicon.svg', type: 'image/svg+xml' },
    ],
    apple: '/favicon.svg',
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
