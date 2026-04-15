import Footer from "../components/Footer";
import Navbar from "../components/Navbar";
import "./globals.css";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "DeepView",
  description: "Visualizing neural networks"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex flex-col min-h-screen">
          <Navbar />
          <main className="flex-grow bg-gray-50 py-4">{children}</main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
