"use client";

import { useEffect, useRef, useState } from "react";
import { MNIST } from "../datasets/load-mnist";
import * as tf from "@tensorflow/tfjs";

const MAX_IMAGES = 25;

export default function Dataset() {
  const trainCanvasRefs = useRef<HTMLCanvasElement[]>(
    Array(MAX_IMAGES).fill(null)
  );
  const testCanvasRefs = useRef<HTMLCanvasElement[]>(
    Array(MAX_IMAGES).fill(null)
  );
  const trainLabelRefs = useRef<HTMLDivElement[]>(Array(MAX_IMAGES).fill(null));
  const testLabelRefs = useRef<HTMLDivElement[]>(Array(MAX_IMAGES).fill(null));
  const [XTrain, setXTrain] = useState<tf.Tensor>();
  const [XTest, setXTest] = useState<tf.Tensor>();
  const [yTrain, setYTrain] = useState<tf.Tensor>();
  const [yTest, setYTest] = useState<tf.Tensor>();

  useEffect(() => {
    (async () => {
      const mnistData = new MNIST();
      await mnistData.load();

      const trainImages = mnistData.getTrain().images;
      const testImages = mnistData.getTest().images;
      const trainLabels = mnistData.getTrain().labels;
      const testLabels = mnistData.getTest().labels;

      setXTrain(trainImages);
      setXTest(testImages);
      setYTrain(trainLabels);
      setYTest(testLabels);
    })();
  }, []);

  useEffect(() => {
    const drawCanvas = (canvasRef: HTMLCanvasElement, oneX: tf.Tensor) => {
      const canvas = canvasRef;
      const ctx = canvas.getContext("2d");

      if (!ctx || !oneX) return;

      const imageSize = 28;
      canvas.width = imageSize;
      canvas.height = imageSize;

      const imageData = new ImageData(imageSize, imageSize);
      const data = oneX.flatten().dataSync();

      for (let i = 0; i < data.length; i++) {
        const pixel = data[i] * 255;
        imageData.data[i * 4 + 0] = pixel;
        imageData.data[i * 4 + 1] = pixel;
        imageData.data[i * 4 + 2] = pixel;
        imageData.data[i * 4 + 3] = 255;
      }

      ctx.putImageData(imageData, 0, 0);
    };

    const setLabel = (divRef: HTMLDivElement, oneY: tf.Tensor) => {
      const div = divRef;
      if (!div || !oneY) return;

      const label = oneY.dataSync()[0];
      div.innerHTML = label.toString();
    };

    const getRandomIndices = (max: number, batchSize: number) => {
      const indices = new Set<number>();
      while (indices.size < batchSize) {
        indices.add(Math.floor(Math.random() * max));
      }
      return Array.from(indices);
    };

    const updateAll = () => {
      if (XTrain && yTrain) {
        const numTrainImages = Math.min(MAX_IMAGES, XTrain.shape[0]);
        const indices = getRandomIndices(XTrain.shape[0], numTrainImages);
        for (let i = 0; i < numTrainImages; i++) {
          const image = XTrain.slice([indices[i], 0, 0, 0], [1, 28, 28, 1]);
          const label = tf.argMax(yTrain.slice([indices[i], 0], [1, 10]), 1);
          drawCanvas(trainCanvasRefs.current[i], image);
          setLabel(trainLabelRefs.current[i], label);
        }
      }
      if (XTest && yTest) {
        const numTestImages = Math.min(MAX_IMAGES, XTest.shape[0]);
        const indices = getRandomIndices(XTest.shape[0], numTestImages);
        for (let i = 0; i < numTestImages; i++) {
          const image = XTest.slice([indices[i], 0, 0, 0], [1, 28, 28, 1]);
          const label = tf.argMax(yTest.slice([indices[i], 0], [1, 10]), 1);
          drawCanvas(testCanvasRefs.current[i], image);
          setLabel(testLabelRefs.current[i], label);
        }
      }
    };

    updateAll();

    // // Update images every 10 seconds
    const interval = setInterval(updateAll, 5000);

    // // Cleanup the interval on component unmount
    return () => clearInterval(interval);
  }, [XTrain, XTest, yTrain, yTest]);

  return (
    <div className="grid grid-cols-2 divide-x justify-between border-dashed border border-gray-300">
      <div className="border-r border-dashed border-gray-300 p-4 text-center">
        <div className="text-xl font-bold mb-2">Train Images</div>

        <div className="flex flex-wrap justify-center">
          {Array.from(
            { length: XTrain ? Math.min(MAX_IMAGES, XTrain.shape[0]) : 0 },
            (_, i) => (
              <div key={i}>
                <div className="w-16 h-16 m-1 p-1 rounded bg-stone-200">
                  <canvas
                    className="w-full max-w-full h-full max-h-full"
                    ref={(el) => {
                      trainCanvasRefs.current[i] = el!;
                    }}
                  />
                </div>
                <div
                  className="pb-2"
                  ref={(el) => {
                    trainLabelRefs.current[i] = el!;
                  }}
                ></div>
              </div>
            )
          )}
        </div>
      </div>
      <div className="p-4 text-center">
        <div className="text-xl font-bold mb-2">Test Images</div>
        <div className="flex flex-wrap justify-center">
          {Array.from(
            { length: XTest ? Math.min(MAX_IMAGES, XTest.shape[0]) : 0 },
            (_, i) => (
              <div key={i}>
                <div className="w-16 h-16 m-1 p-1 rounded bg-stone-200">
                  <canvas
                    className="w-full max-w-full h-full max-h-full"
                    ref={(el) => {
                      testCanvasRefs.current[i] = el!;
                    }}
                  />
                </div>
                <div
                  className="pb-2"
                  ref={(el) => {
                    testLabelRefs.current[i] = el!;
                  }}
                ></div>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
