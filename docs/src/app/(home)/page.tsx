import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="flex flex-col justify-center text-center flex-1">
      <h1 className="text-3xl font-bold mb-2">webtensor</h1>
      <p className="text-lg text-muted-foreground mb-6">
        A tensor library that runs entirely in the browser
      </p>
      <p>
        <Link href="/webtensor/docs" className="font-medium underline hover:no-underline">
          View Documentation →
        </Link>
      </p>
    </div>
  );
}
