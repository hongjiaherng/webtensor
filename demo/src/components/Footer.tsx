import Image from "next/image";

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-white p-4">
      <div className="container mx-auto flex flex-col justify-center items-center">
        <p>
          By <a href="https://github.com/hongjiaherng" className="no-underline hover:underline">jherng</a>.
        </p>
      </div>
    </footer>
  );
}
