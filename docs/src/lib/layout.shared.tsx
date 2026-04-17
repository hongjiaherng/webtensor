import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';
import { gitConfig } from './shared';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export function baseOptions(): BaseLayoutProps {

  return {
    nav: {
      title: (
        <div className="flex items-center h-8">
          <Image
            src={`${basePath}/logo-light.svg`}
            alt="webtensor"
            width={150}
            height={32}
            unoptimized
            priority
            className="dark:hidden h-8 w-auto"
            style={{ width: 'auto', height: 'auto' }}
          />
          <Image
            src={`${basePath}/logo-dark.svg`}
            alt="webtensor"
            width={150}
            height={32}
            unoptimized
            priority
            className="hidden dark:block h-8 w-auto"
            style={{ width: 'auto', height: 'auto' }}
          />
        </div>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
