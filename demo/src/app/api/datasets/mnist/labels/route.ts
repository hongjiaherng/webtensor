import { storage } from "@/firebase/client-app";
import { ref, getDownloadURL } from "firebase/storage";
import { FirebaseError } from "firebase/app";
import { NextRequest, NextResponse } from "next/server";

const TRAIN_LABELS_PATH = "mnist/mnist_labels_train_uint8.dat";
const TEST_LABELS_PATH = "mnist/mnist_labels_test_uint8.dat";

export async function GET(req: NextRequest) {
  // By default, train is true, and source is local
  try {
    const isTrain =
      req.nextUrl.searchParams.get("train")?.toLowerCase() === "false"
        ? false
        : true;
    const isFirebase =
      req.nextUrl.searchParams.get("source")?.toLowerCase() === "firebase";

    const labelsPath = isTrain ? TRAIN_LABELS_PATH : TEST_LABELS_PATH;

    let labelsRes;
    if (isFirebase) {
      const labelsRef = ref(storage, labelsPath);
      const labelsURL = await getDownloadURL(labelsRef);
      labelsRes = await fetch(labelsURL, { cache: "force-cache" });
      console.log("Fetching data from Firebase!");
    } else {
      labelsRes = await fetch(`http://localhost:3000/${labelsPath}`, {
        cache: "no-store"
      });
      console.log("Fetching data from Local!");
    }

    const labelsBuf = await labelsRes.arrayBuffer();

    return new NextResponse(labelsBuf, {
      headers: { "Content-Type": "application/octet-stream" }
    });
  } catch (error: FirebaseError | any) {
    return NextResponse.json(
      { error: error.message, statusCode: 500 },
      { status: 500 }
    );
  }
}
