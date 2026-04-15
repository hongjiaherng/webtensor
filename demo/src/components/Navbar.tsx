export default function Navbar() {
  return (
    <nav className="sticky top-0 bg-gray-900 p-4">
      <div className="container mx-auto flex items-center justify-center">
        <div className="text-white text-center">
          <a href="/">
            <div className="font-bold text-lg">DeepView</div>
            <div>Visualizing the blackbox</div>
          </a>
        </div>
      </div>
    </nav>
  );
}
