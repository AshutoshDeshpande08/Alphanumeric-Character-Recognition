from pathlib import Path
import base64

import cv2
import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "character_model.pth"
WEB_DIR = BASE_DIR / "web"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Handwritten Character Recognition API"
)


# ============================================================
# CNN MODEL
# ============================================================

class CharacterCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 3 * 3,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                36
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
print("Loading character recognition model...")


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]

model = CharacterCNN()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


print("Character model loaded successfully.")
print(f"Classes: {''.join(classes)}")


# ============================================================
# EXTRACT BLUE HANDWRITING
# ============================================================

def extract_blue_ink(image):

    image = cv2.resize(
        image,
        (300, 300),
        interpolation=cv2.INTER_AREA
    )

    blue = image[:, :, 0].astype(
        np.int16
    )

    green = image[:, :, 1].astype(
        np.int16
    )

    red = image[:, :, 2].astype(
        np.int16
    )

    blue_red_difference = (
        blue - red
    )

    blue_green_difference = (
        blue - green
    )

    mask = np.where(
        (
            (blue_red_difference > 5)
            &
            (blue_green_difference > -8)
        ),
        255,
        0
    ).astype(
        np.uint8
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    return mask


# ============================================================
# KEEP MAIN CHARACTER
# ============================================================

def keep_main_character(mask):

    number_of_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if number_of_labels <= 1:
        return None

    largest_label = 1

    largest_area = stats[
        1,
        cv2.CC_STAT_AREA
    ]

    for label in range(
        2,
        number_of_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area > largest_area:

            largest_area = area
            largest_label = label

    if largest_area < 100:
        return None

    character_mask = np.zeros_like(
        mask
    )

    character_mask[
        labels == largest_label
    ] = 255

    return character_mask


# ============================================================
# PREPROCESS CHARACTER
# ============================================================

def preprocess_character(image):

    # --------------------------------------------------------
    # Extract blue handwriting
    # --------------------------------------------------------

    mask = extract_blue_ink(
        image
    )


    # --------------------------------------------------------
    # Keep main character
    # --------------------------------------------------------

    mask = keep_main_character(
        mask
    )

    if mask is None:
        return None


    # --------------------------------------------------------
    # Find character bounding box
    # --------------------------------------------------------

    points = cv2.findNonZero(
        mask
    )

    if points is None:
        return None


    x, y, w, h = cv2.boundingRect(
        points
    )


    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if w < 15 or h < 15:
        return None

    if w > 280 or h > 280:
        return None


    # --------------------------------------------------------
    # Crop character
    # --------------------------------------------------------

    character = mask[
        y:y + h,
        x:x + w
    ]


    # --------------------------------------------------------
    # Make square
    # --------------------------------------------------------

    size = max(
        w,
        h
    )


    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )


    offset_x = (
        size - w
    ) // 2


    offset_y = (
        size - h
    ) // 2


    square[
        offset_y:offset_y + h,
        offset_x:offset_x + w
    ] = character


    # --------------------------------------------------------
    # Add padding
    # --------------------------------------------------------

    padding = max(
        int(size * 0.20),
        4
    )


    square = cv2.copyMakeBorder(
        square,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0
    )


    # --------------------------------------------------------
    # Resize to 28x28
    # --------------------------------------------------------

    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )


    # ========================================================
    # IMPORTANT:
    #
    # EMNIST orientation is different from the camera image.
    #
    # Our diagnostic showed that the correct orientation for
    # the CNN is 90 degrees counter-clockwise.
    # ========================================================

    resized = cv2.rotate(
        resized,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    )


    # --------------------------------------------------------
    # Final binary cleanup
    # --------------------------------------------------------

    _, resized = cv2.threshold(
        resized,
        30,
        255,
        cv2.THRESH_BINARY
    )


    # --------------------------------------------------------
    # Normalize exactly like training
    # --------------------------------------------------------

    normalized = (
        resized.astype(
            np.float32
        ) / 255.0
    )


    normalized = (
        normalized - 0.5
    ) / 0.5


    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    )


    tensor = tensor.unsqueeze(0)

    tensor = tensor.unsqueeze(0)


    return (
        tensor,
        resized
    )


# ============================================================
# PREDICT CHARACTER
# ============================================================

def predict_character(tensor):

    tensor = tensor.to(
        device
    )


    with torch.no_grad():

        output = model(
            tensor
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    predicted_index = (
        prediction.item()
    )


    predicted_character = classes[
        predicted_index
    ]


    confidence = (
        confidence.item()
    )


    return (
        predicted_character,
        confidence
    )


# ============================================================
# IMAGE TO BASE64
# ============================================================

def image_to_base64(image):

    display_image = cv2.resize(
        image,
        (280, 280),
        interpolation=cv2.INTER_NEAREST
    )


    success, encoded = cv2.imencode(
        ".png",
        display_image
    )


    if not success:
        return None


    image_bytes = (
        encoded.tobytes()
    )


    base64_string = (
        base64.b64encode(
            image_bytes
        ).decode("utf-8")
    )


    return (
        "data:image/png;base64,"
        + base64_string
    )


# ============================================================
# API STATUS
# ============================================================

@app.get("/api/status")
def status():

    return {
        "status": "running",
        "model": "character_cnn",
        "classes": classes,
        "device": str(device)
    }


# ============================================================
# PROCESS IMAGE
# ============================================================

@app.post("/api/process-image")
async def process_image(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image_bytes = (
        await file.read()
    )


    if not image_bytes:

        return {
            "success": False,
            "error": "Empty image received."
        }


    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )


    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if image is None:

        return {
            "success": False,
            "error": "Could not decode image."
        }


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    result = preprocess_character(
        image
    )


    if result is None:

        return {
            "success": False,
            "error": (
                "Could not isolate the "
                "handwritten character."
            )
        }


    tensor, processed_image = result


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    character, confidence = (
        predict_character(
            tensor
        )
    )


    # --------------------------------------------------------
    # Diagnostic image
    # --------------------------------------------------------

    processed_image_base64 = (
        image_to_base64(
            processed_image
        )
    )


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "success": True,

        "character": character,

        "confidence": round(
            confidence,
            4
        ),

        "confidence_percent": round(
            confidence * 100,
            2
        ),

        "processed_image":
            processed_image_base64
    }


# ============================================================
# SERVE FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=WEB_DIR,
        html=True
    ),
    name="web"
)