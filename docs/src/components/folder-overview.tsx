import { Card, Cards } from 'fumadocs-ui/components/card';
import { findPath } from 'fumadocs-core/page-tree';
import type { Folder, Item, Node, Root } from 'fumadocs-core/page-tree';
import { source } from '@/lib/source';

function childLink(node: Node): { href: string; title: string; description?: string } | null {
  if (node.type === 'page') {
    return {
      href: (node as Item).url,
      title: String(node.name),
      description: node.description ? String(node.description) : undefined,
    };
  }
  if (node.type === 'folder') {
    const folder = node as Folder;
    const target = folder.index ?? firstPageIn(folder.children);
    if (!target) return null;
    return {
      href: target.url,
      title: String(folder.name),
      description: folder.description ? String(folder.description) : undefined,
    };
  }
  return null;
}

function firstPageIn(nodes: Node[]): { url: string } | null {
  for (const node of nodes) {
    if (node.type === 'page') return node as Item;
    if (node.type === 'folder') {
      const nested = firstPageIn((node as Folder).children);
      if (nested) return nested;
    }
  }
  return null;
}

// Find the Folder node in the tree whose children contain a page with the
// given URL. Returns both the folder and the matching child index.
function findParentFolder(nodes: Node[], url: string): Folder | null {
  // Try findPath first — works for non-root folder index pages.
  const path = findPath(nodes, (n) => n.type === 'page' && (n as Item).url === url);
  if (path) {
    const folder = path.findLast((n) => n.type === 'folder') as Folder | undefined;
    if (folder) return folder;
  }

  // Fallback: root-marked folders don't have .index set, so findPath may not
  // traverse into them via the index shortcut. Search children directly.
  for (const node of nodes) {
    if (node.type !== 'folder') continue;
    const folder = node as Folder;
    const hasPage = folder.children.some((c) => c.type === 'page' && (c as Item).url === url);
    if (hasPage) return folder;
    const nested = findParentFolder(folder.children, url);
    if (nested) return nested;
  }
  return null;
}

export function FolderOverview({ url }: { url: string }) {
  const tree = source.getPageTree() as Root;

  // Root-marked folders (like `api/`) end up in `tree.fallback` when they
  // aren't explicitly listed in the parent `meta.json`'s pages array — so we
  // must search both trees. `TreeContextProvider` does the same.
  const parentFolder =
    findParentFolder(tree.children, url) ??
    (tree.fallback ? findParentFolder(tree.fallback.children, url) : null);

  if (!parentFolder) return null;

  const cards = parentFolder.children
    .filter((n) => n.type !== 'separator' && !(n.type === 'page' && (n as Item).url === url))
    .map(childLink)
    .filter((c): c is NonNullable<typeof c> => c !== null);

  if (cards.length === 0) return null;

  return (
    <Cards>
      {cards.map((c) => (
        <Card key={c.href} href={c.href} title={c.title} description={c.description} />
      ))}
    </Cards>
  );
}
