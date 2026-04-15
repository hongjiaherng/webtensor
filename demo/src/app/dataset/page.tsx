import DatasetViz from "@/components/DatasetViz";

export default function Page() {
  return (
    <div className="container mx-auto bg-white shadow-sm rounded-md border-dashed divide-y divide-dashed">
      <div className="p-6">
        <div className="text-3xl font-bold mb-2">Datasets</div>
        <div className="text-gray-600">
          Gallery of datasets that are available for use.
        </div>
      </div>
      <div className="p-6">
        <div className="text-2xl font-bold mb-2">MNIST</div>
        <div className="text-gray-500 mb-2">Handwritten digits</div>
        <DatasetViz />
      </div>
    </div>
  );
}
