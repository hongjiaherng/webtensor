import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';
import { gitConfig } from './shared';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <div className="relative h-7 w-[120px] sm:w-[150px]">
          <Image
            src={`${basePath}/logo-light.svg`}
            alt="webtensor"
            fill
            unoptimized
            priority
            className="dark:hidden object-contain object-left"
          />
          <Image
            src={`${basePath}/logo-dark.svg`}
            alt="webtensor"
            fill
            unoptimized
            priority
            className="hidden dark:block object-contain object-left"
          />
        </div>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
