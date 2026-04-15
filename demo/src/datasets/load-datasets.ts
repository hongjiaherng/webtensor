// TODO:
// - [ ] reset offset when all instances are used (when do i known all instances are used?)
// - [ ] add in CIFAR10 dataset

import * as tf from "@tensorflow/tfjs";

type DatasetSource = "local" | "firebase";

export class MNISTDataset {
  readonly nW: number = 28;
  readonly nH: number = 28;
  readonly nC: number = 1;
  readonly nClasses: number = 10;
  readonly train: boolean = true;
  readonly batchSize: number = 32;
  readonly shuffle: boolean = true;
  readonly source: DatasetSource = "local";
  imagesData: Float32Array | undefined;
  labelsData: Uint8Array | undefined;
  nInstances: number = 0;

  private _shuffledIndices: Uint32Array | undefined;
  private _offset: number = 0;
  private _nextInstanceIdx: () => number;

  private constructor(
    train: boolean,
    source: DatasetSource,
    batchSize?: number,
    shuffle?: boolean
  ) {
    this.train = train;
    this.source = source;
    this.batchSize = batchSize || this.batchSize;
    this.shuffle = shuffle === undefined ? this.shuffle : shuffle;

    this._nextInstanceIdx = this.shuffle
      ? () => {
          const instance_idx = this._shuffledIndices![this._offset];
          this._offset++;
          return instance_idx;
        }
      : () => {
          const instance_idx = this._offset;
          this._offset++;
          return instance_idx;
        };
  }

  public static async load(
    train: boolean,
    source: DatasetSource,
    batchSize?: number,
    shuffle?: boolean
  ) {
    const dataset = new MNISTDataset(train, source, batchSize, shuffle);

    // fetch images and labels in array form
    const [imagesData, labelsData] = await Promise.all([
      dataset.fetchImages(),
      dataset.fetchLabels()
    ]);
    // set number of instances, imagesData, labelsData
    dataset.imagesData = imagesData;
    dataset.labelsData = labelsData;
    dataset.nInstances =
      dataset.imagesData.length / (dataset.nW * dataset.nH * dataset.nC);

    // create shuffle indices if shuffle is true
    if (dataset.shuffle) {
      dataset._shuffledIndices = tf.util.createShuffledIndices(
        dataset.nInstances
      );
    }

    return dataset;
  }

  private async fetchImages(): Promise<Float32Array> {
    const url = `http://localhost:3000/api/datasets/mnist/images?source=${this.source}&train=${this.train}`;
    const res = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "image/png"
      },
      cache: "no-store"
    });

    if (!res.ok) {
      throw new Error("Failed to fetch images");
    }

    const blob = await res.blob();

    return new Promise<Float32Array>((resolve, reject) => {
      const imgElement = new Image();
      imgElement.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = imgElement.width;
        canvas.height = imgElement.height;

        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);

        const imgElementData = ctx.getImageData(
          0,
          0,
          canvas.width,
          canvas.height
        );
        const imagesData = new Float32Array(imgElementData.data.length / 4);

        for (let i = 0; i < imgElementData.data.length; i += 4) {
          imagesData[i / 4] = imgElementData.data[i] / 255; // normalize [0, 255] to [0, 1]
        }

        resolve(imagesData);
      };
      imgElement.src = URL.createObjectURL(blob);
    });
  }

  private async fetchLabels(): Promise<Uint8Array> {
    const url = `http://localhost:3000/api/datasets/mnist/labels?source=${this.source}&train=${this.train}`;
    const res = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/octet-stream"
      },
      cache: "no-store"
    });
    const buf = await res.arrayBuffer();
    const labelsData = new Uint8Array(buf);
    return labelsData;
  }

  public nextBatch() {
    const batchImagesData = new Float32Array(
      this.batchSize * this.nW * this.nH * this.nC
    );
    const batchLabelsData = new Uint8Array(this.batchSize);

    for (let i = 0; i < this.batchSize; i++) {
      const instance_idx = this._nextInstanceIdx();

      batchImagesData.set(
        this.imagesData!.slice(
          instance_idx * this.nW * this.nH * this.nC,
          (instance_idx + 1) * this.nW * this.nH * this.nC
        ),
        i * this.nW * this.nH * this.nC
      );
      batchLabelsData.fill(this.labelsData![instance_idx], i, i + 1);
    }

    const xs = tf.tensor4d(
      batchImagesData,
      [this.batchSize, this.nW, this.nH, this.nC],
      "float32"
    );
    const ys = tf.oneHot(tf.tensor1d(batchLabelsData), this.nClasses);

    return { xs, ys };
  }
}
