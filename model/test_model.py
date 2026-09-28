import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# --------------------------------------------------
# 2. Same CNN architecture used during training
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Model path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "character_model.pth"


# --------------------------------------------------
# 4. Load trained model
# --------------------------------------------------

print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]

model = CharacterCNN()
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

print("Model loaded successfully.")

print(f"Number of classes: {len(classes)}")
print(f"Classes: {''.join(classes)}")


# --------------------------------------------------
# 5. Same preprocessing used during training
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# --------------------------------------------------
# 6. Load EMNIST test dataset
# --------------------------------------------------

print("\nLoading EMNIST test dataset...")

dataset = datasets.EMNIST(
    root=BASE_DIR / "data" / "dataset",
    split="balanced",
    train=False,
    download=True,
    transform=transform
)

print(f"Total test samples: {len(dataset)}")


# --------------------------------------------------
# 7. Create mapping from original EMNIST labels
#    to our 36-class model labels
# --------------------------------------------------

target_characters = set(classes)

target_indices = []

for index, character in enumerate(dataset.classes):
    if character in target_characters:
        target_indices.append(index)


label_to_model_index = {
    original_index: model_index
    for model_index, original_index in enumerate(target_indices)
}


# --------------------------------------------------
# 8. Test the model
# --------------------------------------------------

correct = 0
total = 0

max_test_samples = 1000

print("\nRunning model test...")
print("-" * 60)

sample_predictions = []


with torch.no_grad():

    for dataset_index in range(len(dataset)):

        if total >= max_test_samples:
            break

        image, original_label = dataset[dataset_index]

        # Ignore lowercase classes that are not part
        # of our 36-class target
        if original_label not in label_to_model_index:
            continue

        image = image.unsqueeze(0).to(device)

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

        predicted_index = predicted_index.item()
        confidence = confidence.item()

        actual_character = dataset.classes[original_label]
        predicted_character = classes[predicted_index]

        if predicted_character == actual_character:
            correct += 1

        total += 1

        # Store first 20 predictions for display
        if len(sample_predictions) < 20:
            sample_predictions.append(
                (
                    actual_character,
                    predicted_character,
                    confidence
                )
            )


# --------------------------------------------------
# 9. Display sample predictions
# --------------------------------------------------

print("\nSample Predictions")
print("-" * 60)

for number, (actual, predicted, confidence) in enumerate(
    sample_predictions,
    start=1
):

    status = "✓" if actual == predicted else "✗"

    print(
        f"{number:02d}. "
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"Confidence: {confidence * 100:.2f}% "
        f"{status}"
    )


# --------------------------------------------------
# 10. Accuracy
# --------------------------------------------------

accuracy = (correct / total) * 100

print("\n" + "=" * 60)

print(f"Test samples evaluated: {total}")
print(f"Correct predictions:    {correct}")
print(f"Incorrect predictions:  {total - correct}")
print(f"Test accuracy:          {accuracy:.2f}%")

print("=" * 60)

print("\nModel test completed.")