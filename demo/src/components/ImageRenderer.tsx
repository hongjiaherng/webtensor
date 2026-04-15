"use client";
import { useEffect, useRef } from "react";

export default function ImageRenderer({
  arr,
  w,
  h,
  c = 1
}: {
  arr: Uint8ClampedArray;
  w: number;
  h: number;
  c?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const imageData = ctx.createImageData(w, h);

    if (c === 1) {
      for (let i = 0; i < arr.length; i++) {
        imageData.data[i * 4] = arr[i];
        imageData.data[i * 4 + 1] = arr[i];
        imageData.data[i * 4 + 2] = arr[i];
        imageData.data[i * 4 + 3] = 255;
      }
    } else if (c === 3) {
      throw new Error("Not implemented");
    } else {
      throw new Error("Invalid value for 'c'. Only 1 or 3 are allowed.");
    }

    ctx.putImageData(imageData, 0, 0);
  }, [arr, h, w, c]);

  return <canvas ref={canvasRef} width={w} height={h}></canvas>;
}
