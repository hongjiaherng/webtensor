import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';
import { appName, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <div className="flex items-center h-8">
          <Image
            src="/logo-light.svg"
            alt="webtensor"
            width={150}
            height={32}
            unoptimized
            priority
            className="dark:hidden h-8 w-auto"
          />
          <Image
            src="/logo-dark.svg"
            alt="webtensor"
            width={150}
            height={32}
            unoptimized
            priority
            className="hidden dark:block h-8 w-auto"
          />
        </div>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
