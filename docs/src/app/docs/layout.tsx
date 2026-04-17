import { source } from '@/lib/source';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { baseOptions } from '@/lib/layout.shared';

export default function Layout({ children }: LayoutProps<'/docs'>) {
  return (
    <DocsLayout
      tree={source.getPageTree()}
      containerProps={{ className: '[--fd-layout-width:9999px]' }}
      {...baseOptions()}
    >
      {children}
    </DocsLayout>
  );
}
