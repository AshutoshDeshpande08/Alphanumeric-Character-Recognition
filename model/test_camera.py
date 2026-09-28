import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


# ============================================================
# 1. CNN MODEL
# ============================================================

class CharacterCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 36)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "character_model.pth"


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("Loading trained model...")

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

print("Model loaded successfully.")
print(f"Classes: {''.join(classes)}")


# ============================================================
# 5. PREPROCESS CHARACTER
# ============================================================

def preprocess_character(cropped):
    """
    Convert a cropped camera image of a handwritten
    character into a 28x28 image suitable for the CNN.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(
        cropped,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur to remove camera noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Convert handwriting to white and background to black
    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Find the largest contour
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # Ignore extremely small detections
    if w < 10 or h < 10:
        return None

    # Crop the actual character
    character = threshold[
        y:y + h,
        x:x + w
    ]

    # Make a square canvas
    size = max(w, h)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    # Center character inside square
    offset_x = (size - w) // 2
    offset_y = (size - h) // 2

    square[
        offset_y:offset_y + h,
        offset_x:offset_x + w
    ] = character

    # Add some padding
    padding = int(size * 0.20)

    square = cv2.copyMakeBorder(
        square,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Resize to model input size
    resized = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # Convert 0-255 to 0-1
    normalized = resized.astype(
        np.float32
    ) / 255.0

    # Match training normalization:
    # Normalize((0.5,), (0.5,))
    normalized = (
        normalized - 0.5
    ) / 0.5

    # Convert to PyTorch tensor
    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    )

    # Shape:
    # 28x28
    #   ↓
    # 1x28x28
    #   ↓
    # 1x1x28x28
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor, resized


# ============================================================
# 6. PREDICT CHARACTER
# ============================================================

def predict_character(tensor):

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = prediction.item()

    predicted_character = classes[
        predicted_index
    ]

    confidence = confidence.item()

    return predicted_character, confidence


# ============================================================
# 7. CAMERA
# ============================================================

print("\nStarting camera...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("\nERROR: Could not open camera.")

    print(
        "If you are running this inside GitHub Codespaces, "
        "the Python process cannot directly access your "
        "computer's webcam."
    )

    print(
        "We will connect the browser camera to the model "
        "later through FastAPI."
    )

    exit()


print("\nCamera started.")
print("Place ONE handwritten character inside the box.")
print("Press SPACE to capture.")
print("Press Q to quit.")


# ============================================================
# 8. CAMERA LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    height, width = frame.shape[:2]

    # Detection area in the center
    box_size = min(
        300,
        height - 40,
        width - 40
    )

    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2

    x2 = x1 + box_size
    y2 = y1 + box_size

    # Draw capture box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Place ONE character inside box",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Capture | Q = Quit",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Handwritten Character Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Capture
    if key == 32:

        roi = frame[
            y1:y2,
            x1:x2
        ]

        result = preprocess_character(roi)

        if result is None:

            print(
                "\nCould not detect a character."
            )

            continue

        tensor, processed_image = result

        character, confidence = predict_character(
            tensor
        )

        print("\n" + "=" * 50)

        print(
            f"Prediction: {character}"
        )

        print(
            f"Confidence: {confidence * 100:.2f}%"
        )

        print("=" * 50)

        # Show processed 28x28 character
        display_image = cv2.resize(
            processed_image,
            (280, 280),
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(
            "Processed Character",
            display_image
        )


# ============================================================
# 9. CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("\nCamera test finished.")