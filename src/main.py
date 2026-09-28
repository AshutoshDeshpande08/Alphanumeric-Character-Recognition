from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np


app = FastAPI(title="VisionOCR")


# ==========================================
# API STATUS
# ==========================================

@app.get("/api/status")
def status():

    return {
        "project": "VisionOCR",
        "status": "running"
    }


# ==========================================
# IMAGE PROCESSING
# ==========================================

@app.post("/api/process-image")
async def process_image(file: UploadFile = File(...)):

    # --------------------------------------
    # 1. Read uploaded image
    # --------------------------------------

    image_bytes = await file.read()


    # --------------------------------------
    # 2. Convert bytes to NumPy array
    # --------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )


    # --------------------------------------
    # 3. Decode image with OpenCV
    # --------------------------------------

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if image is None:

        return {
            "success": False,
            "message": "Could not decode image"
        }


    # --------------------------------------
    # 4. Get image dimensions
    # --------------------------------------

    height, width, channels = image.shape


    # --------------------------------------
    # 5. Convert to grayscale
    # --------------------------------------

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------
    # 6. Apply Gaussian blur
    # --------------------------------------

    blurred = cv2.GaussianBlur(
        grayscale,
        (5, 5),
        0
    )


    # --------------------------------------
    # 7. Apply threshold
    # --------------------------------------

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    # --------------------------------------
    # 8. Calculate basic image statistics
    # --------------------------------------

    brightness = float(
        np.mean(grayscale)
    )


    contrast = float(
        np.std(grayscale)
    )


    # --------------------------------------
    # 9. Terminal information
    # --------------------------------------

    print(
        f"Received image: {width}x{height}"
    )

    print(
        f"Brightness: {brightness:.2f}"
    )

    print(
        f"Contrast: {contrast:.2f}"
    )


    # --------------------------------------
    # 10. Return analysis
    # --------------------------------------

    return {

        "success": True,

        "message": "Image processed successfully",

        "image": {
            "width": width,
            "height": height,
            "channels": channels
        },

        "processing": {
            "grayscale": True,
            "blur": True,
            "threshold": True
        },

        "analysis": {
            "brightness": round(
                brightness,
                2
            ),

            "contrast": round(
                contrast,
                2
            )
        }
    }


# ==========================================
# FRONTEND
# ==========================================

app.mount(
    "/",
    StaticFiles(
        directory="web",
        html=True
    ),
    name="web"
)