'use client';
import SearchDialog from '@/components/search';
import { RootProvider } from 'fumadocs-ui/provider/next';
import { type ReactNode, useEffect } from 'react';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export function Provider({ children }: { children: ReactNode }) {
  useEffect(() => {
    if (!basePath) return;
    const orig = window.fetch;
    // `typeof fetch` carries a `preconnect` static in newer @types/node/bun —
    // reuse the original as the prototype so static properties flow through.
    const wrapped = Object.assign(
      (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const url =
          typeof input === 'string'
            ? input
            : input instanceof URL
              ? input.toString()
              : (input as Request).url;
        // Absolute in-app paths (e.g. `/api/search`, `/llms.mdx/...`) need
        // basePath prefixed — raw fetch() doesn't do it like next/link does.
        // Skip protocol-relative (`//`), already-prefixed URLs, and absolute
        // URLs with a scheme (handled above via URL/Request branch).
        if (
          typeof input === 'string' &&
          url.startsWith('/') &&
          !url.startsWith('//') &&
          !url.startsWith(basePath + '/')
        ) {
          return orig(basePath + url, init);
        }
        return orig(input, init);
      },
      { preconnect: (orig as { preconnect?: (url: string) => void }).preconnect?.bind(orig) },
    ) as typeof fetch;
    window.fetch = wrapped;
    return () => {
      window.fetch = orig;
    };
  }, []);

  return <RootProvider search={{ SearchDialog }}>{children}</RootProvider>;
}
