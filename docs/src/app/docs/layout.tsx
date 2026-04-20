import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions } from '@/lib/layout.shared';
import { docsRoute } from '@/lib/shared';
import type { Folder } from 'fumadocs-core/page-tree';
import { BookOpenText, Braces } from 'lucide-react';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  const tree = source.getPageTree();

  // Find the api folder for $folder-based active-tab detection. Root-marked
  // folders not listed in the parent meta.json pages end up under tree.fallback.
  const findRoot = (nodes: typeof tree.children): Folder | undefined =>
    nodes.find((n): n is Folder => n.type === 'folder' && (n as Folder).root === true);
  const apiFolder = findRoot(tree.children) ?? (tree.fallback ? findRoot(tree.fallback.children) : undefined);

  return (
    <DocsLayout
      tree={tree}
      containerProps={{ className: '[--fd-layout-width:9999px]' }}
      tabs={[
        {
          title: 'Documentation',
          url: docsRoute,
          icon: (
            <div className="flex size-full items-center justify-center text-fd-primary">
              <BookOpenText className="size-5" />
            </div>
          ),
        },
        {
          title: 'API Reference',
          url: `${docsRoute}/api`,
          icon: (
            <div className="flex size-full items-center justify-center text-fd-primary">
              <Braces className="size-5" />
            </div>
          ),
          ...(apiFolder ? { $folder: apiFolder } : {}),
        },
      ]}
      {...baseOptions()}
    >
      {children}
    </DocsLayout>
  );
}
