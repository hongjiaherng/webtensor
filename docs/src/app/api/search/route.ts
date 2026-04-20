import { source } from '@/lib/source';
import { createFromSource } from 'fumadocs-core/search/server';
import { findPath, type Node } from 'fumadocs-core/page-tree';
import { basename, extname } from 'node:path';

export const revalidate = false;

// Fumadocs' default breadcrumb builder walks `tree.children` only. Pages under
// `root: true` folders (our `api/` reference tab) live in `tree.fallback`
// instead, so those results come back with no breadcrumbs. Provide a custom
// `buildIndex` that searches both trees.
function breadcrumbsFor(url: string): string[] | undefined {
  const tree = source.getPageTree();
  const predicate = (n: Node) => n.type === 'page' && n.url === url;
  const path =
    findPath(tree.children, predicate) ??
    (tree.fallback ? findPath(tree.fallback.children, predicate) : null);
  if (!path) return undefined;
  path.pop();

  const labels: string[] = [];

  // Check if the current page is part of the API Reference
  const isApiReference = path.some((segment) => url.includes('/docs/api/'));
  if (typeof tree.name === 'string' && tree.name && !isApiReference) labels.push(tree.name);
  for (const segment of path) {
    if (typeof segment.name === 'string' && segment.name) labels.push(segment.name);
  }
  console.log(`breadcrumbs for ${url}: ${labels.join(' > ')}`);
  return labels;
}

export const { staticGET: GET } = createFromSource(source, {
  // https://docs.orama.com/docs/orama-js/supported-languages
  language: 'english',
  async buildIndex(page) {
    const data = page.data as {
      title?: string;
      description?: string;
      structuredData?: unknown | (() => Promise<unknown>);
      load?: () => Promise<{ structuredData: unknown }>;
    };
    const structuredData =
      typeof data.structuredData === 'function'
        ? await data.structuredData()
        : (data.structuredData ?? (data.load ? (await data.load()).structuredData : undefined));
    if (!structuredData) {
      throw new Error(`Cannot find structured data from page ${page.url}`);
    }
    return {
      title: data.title ?? basename(page.path, extname(page.path)),
      description: data.description,
      url: page.url,
      id: page.url,
      structuredData: structuredData as never,
      breadcrumbs: breadcrumbsFor(page.url),
    };
  },
});
