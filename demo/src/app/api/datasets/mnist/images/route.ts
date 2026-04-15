import { storage } from "@/firebase/client-app";
import { ref, getDownloadURL } from "firebase/storage";
import { NextRequest, NextResponse } from "next/server";
import { FirebaseError } from "firebase/app";

const TRAIN_IMAGES_PATH = "mnist/mnist_images_train.png";
const TEST_IMAGES_PATH = "mnist/mnist_images_test.png";

export async function GET(req: NextRequest) {
  // By default, train is true, and source is local
  try {
    const isTrain =
      req.nextUrl.searchParams.get("train")?.toLowerCase() === "false"
        ? false
        : true;
    const isFirebase =
      req.nextUrl.searchParams.get("source")?.toLowerCase() === "firebase";

    const imagesPath = isTrain ? TRAIN_IMAGES_PATH : TEST_IMAGES_PATH;

    let imagesRes;
    if (isFirebase) {
      const imagesRef = ref(storage, imagesPath);
      const imagesURL = await getDownloadURL(imagesRef);
      imagesRes = await fetch(imagesURL, { cache: "no-store" });
      console.log("Fetching data from Firebase!");
    } else {
      imagesRes = await fetch(`http://localhost:3000/${imagesPath}`, {
        cache: "no-store"
      });
      console.log("Fetching data from Local!");
    }

    const imagesBuf = await imagesRes.arrayBuffer();
    const imagesBlob = new Blob([imagesBuf], { type: "image/png" });

    return new NextResponse(imagesBlob, {
      headers: { "Content-Type": "image/png" }
    });
  } catch (error: FirebaseError | any) {
    return NextResponse.json(
      { error: error.message, statusCode: 500 },
      { status: 500 }
    );
  }
}
