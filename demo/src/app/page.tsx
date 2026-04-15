import Dataset from "@/components/Dataset1";

export default async function Home() {
  return (
    <div className="flex flex-col">
      <div className="container mx-auto bg-white rounded-lg shadow-sm border-dashed border-gray-300 divide-y divide-dashed">
        <div className="p-6">
          <div className="text-3xl font-bold mb-2">ConvNet</div>
          <div className="text-base">
            Train a ConvNet to classify MNIST digits.
          </div>
        </div>
        <div className="p-6">
          <div className="text-xl font-bold mb-2">Description</div>
          <div className="text-base">
            Lorem ipsum dolor sit amet consectetur adipisicing elit. Et, quasi
            cupiditate voluptas beatae ducimus voluptatibus hic sapiente
            nesciunt similique, ullam repudiandae tenetur dolore maxime adipisci
            veritatis fuga repellat quidem atque quibusdam ex commodi. Deleniti,
            fugiat facilis animi corrupti repellendus expedita esse aliquid
            magnam ut ducimus eius labore! Id fugit natus perspiciatis expedita
            nesciunt, tempora atque ratione commodi, obcaecati iusto cumque est,
            molestiae tempore unde officiis.
          </div>
        </div>
        <div className="p-6">
          <div className="text-xl font-bold mb-2">Training Stats</div>
          <div>
            Lorem ipsum dolor sit amet consectetur adipisicing elit. Et, quasi
            cupiditate voluptas beatae ducimus voluptatibus hic sapiente
            nesciunt similique, ullam repudiandae tenetur dolore maxime adipisci
            veritatis fuga repellat quidem atque quibusdam ex commodi. Deleniti,
            fugiat facilis animi corrupti repellendus expedita esse aliquid
            magnam ut ducimus eius labore! Id fugit natus perspiciatis expedita
            nesciunt, tempora atque ratione commodi, obcaecati iusto cumque est,
            molestiae tempore unde officiis.
          </div>
        </div>
        <div className="p-6">
          <div className="text-xl font-bold mb-2">Network Visualization</div>
        </div>
        <div className="p-6">
          <div className="text-xl font-bold mb-2">Dataset Visualization</div>
          {/* <Dataset /> */}
        </div>
      </div>
    </div>
  );
}
