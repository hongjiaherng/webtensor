"use client";
import { use, useEffect, useState } from "react";
import ImageRenderer from "./ImageRenderer";
import { MNISTDataset } from "@/datasets/load-datasets";
import * as tf from "@tensorflow/tfjs";

const fetchLabels = async () => {
  const labelsRes = await fetch(
    "http://localhost:3000/api/datasets/mnist/labels?train=true&source=local",
    {
      method: "GET",
      headers: {
        "Content-Type": "application/octet-stream"
      },
      cache: "no-store"
    }
  );
  const labelsBuf = await labelsRes.arrayBuffer();
  const labels = new Uint8Array(labelsBuf);
  return labels;
};

const fetchImages = async () => {
  const imagesRes = await fetch(
    "http://localhost:3000/api/datasets/mnist/images?train=true&source=local",
    {
      method: "GET",
      headers: {
        "Content-Type": "image/png"
      },
      cache: "no-store"
    }
  );

  const imagesBlob = await imagesRes.blob();

  return new Promise<Uint8ClampedArray>((resolve, reject) => {
    const imgElement = new Image();
    imgElement.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = imgElement.width;
      canvas.height = imgElement.height;

      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(imgElement, 0, 0);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const imagesArr = new Uint8ClampedArray(imageData.data.length / 4);
      for (let i = 0; i < imageData.data.length; i += 4) {
        imagesArr[i / 4] = imageData.data[i];
      }

      resolve(imagesArr);
    };
    imgElement.src = URL.createObjectURL(imagesBlob);
  });
};

export default function DatasetViz() {
  const [batchXs, setBatchXs] = useState<tf.Tensor>();
  const [batchYs, setBatchYs] = useState<tf.Tensor>();

  useEffect(() => {
    (async () => {
      const datasetLoader = await MNISTDataset.load(false, "local", 32, true);
      const { xs, ys } = datasetLoader.nextBatch();
      setBatchXs(xs);
      setBatchYs(ys);
    })();
  }, []);

  useEffect(() => {
    if (!batchXs) return;
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext("2d")!;
    const imageData = new ImageData(28, 28);
    const data = batchXs.flatten().dataSync();
    for (let i = 0; i < data.length; i++) {
      const pixel = data[i] * 255;
      imageData.data[i * 4 + 0] = pixel;
      imageData.data[i * 4 + 1] = pixel;
      imageData.data[i * 4 + 2] = pixel;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    document.body.appendChild(canvas);
  }, [batchXs]);

  return <div></div>;
}
