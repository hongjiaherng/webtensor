'use client';
import SearchDialog from '@/components/search';
import { RootProvider } from 'fumadocs-ui/provider/next';
import { type ReactNode, useEffect } from 'react';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export function Provider({ children }: { children: ReactNode }) {
  useEffect(() => {
    if (!basePath) return;
    const orig = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : (input as Request).url;
      if (url.startsWith('/api/')) return orig(basePath + url, init);
      return orig(input, init);
    };
    return () => {
      window.fetch = orig;
    };
  }, []);

  return <RootProvider search={{ SearchDialog }}>{children}</RootProvider>;
}
