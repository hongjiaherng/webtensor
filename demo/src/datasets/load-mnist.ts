import * as tf from "@tensorflow/tfjs";

const MNIST_IMAGES_PATH = "/api/dataset/mnist/images";
const MNIST_LABELS_PATH = "/api/dataset/mnist/labels";

const IMAGES_H = 28;
const IMAGES_W = 28;
const IMAGES_C = 1;
const N_TRAIN = 60000;
const N_TEST = 10000;
const N_CLASSES = 10;

const fetchImages2 = async (datasetType: string) => {
  const n_samples = datasetType === "train" ? N_TRAIN : N_TEST;

  const imagesRes = await fetch(
    `http://localhost:3000${MNIST_IMAGES_PATH}/${datasetType}`,
    {
      method: "GET",
      headers: {
        "Content-Type": "image/png"
      },
      cache: "no-store"
    }
  );

  if (!imagesRes.ok) {
    throw new Error("Failed to fetch images");
  }

  const imagesBlob = await imagesRes.blob();
  const imagesUrl = URL.createObjectURL(imagesBlob);

  const img = new Image();
  const canvas = document.createElement("canvas");

  // Set willReadFrequently to true before accessing imageData
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const imgRequest = new Promise<Float32Array>((resolve, reject) => {
    img.crossOrigin = "";
    img.onload = () => {
      img.width = img.naturalWidth;
      img.height = img.naturalHeight;

      const datasetBytesBuffer = new ArrayBuffer(
        n_samples * IMAGES_H * IMAGES_W * IMAGES_C * 4
      );

      const chunkSize = 10000;
      canvas.width = img.width;
      canvas.height = chunkSize;

      for (let i = 0; i < n_samples / chunkSize; i++) {
        const datasetBytesView = new Float32Array(
          datasetBytesBuffer,
          i * IMAGES_H * IMAGES_W * chunkSize * 4,
          IMAGES_H * IMAGES_W * chunkSize
        );
        ctx.drawImage(
          img,
          0,
          i * chunkSize,
          img.width,
          chunkSize,
          0,
          0,
          img.width,
          chunkSize
        );

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        for (let j = 0; j < imageData.data.length / 4; j++) {
          // All channels hold an equal value since the image is grayscale, so
          // just read the red channel.
          datasetBytesView[j] = imageData.data[j * 4] / 255;
        }
      }
      const datasetImages = new Float32Array(datasetBytesBuffer);

      resolve(datasetImages);
    };
    img.src = imagesUrl;
  });

  return imgRequest;
};

const fetchImages = async (datasetType: string) => {
  try {
    const imagesRes = await fetch(
      `http://localhost:3000${MNIST_IMAGES_PATH}/${datasetType}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "image/png"
        },
        cache: "no-store"
      }
    );

    if (!imagesRes.ok) {
      throw new Error("Failed to fetch images");
    }

    const imagesBlob = await imagesRes.blob();

    return new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader();

      reader.onloadend = () => {
        if (reader.readyState === FileReader.DONE) {
          const arrayBuffer = reader.result as ArrayBuffer;
          resolve(arrayBuffer);
        } else {
          reject(new Error("Failed to convert image to array buffer"));
        }
      };

      reader.onerror = () => {
        reject(new Error("Failed to read image data"));
      };

      reader.readAsArrayBuffer(imagesBlob);
    });
  } catch (error) {
    throw error;
  }
};

const fetchLabels = async (datasetType: string) => {
  const labelsRes = await fetch(
    `http://localhost:3000${MNIST_LABELS_PATH}/${datasetType}`,
    {
      method: "GET",
      headers: {
        "Content-Type": "application/octet-stream"
      },
      cache: "force-cache"
    }
  );
  const labelsBuf = await labelsRes.arrayBuffer();
  const datasetLabels = new Uint8Array(labelsBuf);
  return datasetLabels;
};

export class MNIST {
  trainImages?: ArrayBuffer;
  trainLabels?: ArrayBuffer;
  testImages?: ArrayBuffer;
  testLabels?: ArrayBuffer;

  constructor() {}

  async load() {
    let [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
      fetchImages2("train"),
      fetchLabels("train"),
      fetchImages2("test"),
      fetchLabels("test")
    ]);

    this.trainImages = trainImages;
    this.testImages = testImages;
    this.trainLabels = trainLabels;
    this.testLabels = testLabels;
  }

  getTrain(): { images: tf.Tensor; labels: tf.Tensor } {
    return {
      images: tf.tensor4d(
        new Float32Array(this.trainImages!),
        [N_TRAIN, IMAGES_H, IMAGES_W, IMAGES_C],
        "float32"
      ),
      labels: tf.oneHot(
        tf.tensor1d(new Uint8Array(this.trainLabels!), "int32"),
        N_CLASSES
      )
    };
  }

  getTest(): { images: tf.Tensor; labels: tf.Tensor } {
    return {
      images: tf.tensor4d(
        new Float32Array(this.testImages!),
        [N_TEST, IMAGES_H, IMAGES_W, IMAGES_C],
        "float32"
      ),
      labels: tf.oneHot(
        tf.tensor1d(new Uint8Array(this.testLabels!), "int32"),
        N_CLASSES
      )
    };
  }
}
