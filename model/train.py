from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

MODEL_DIR = PROJECT_ROOT / "model"

MODEL_PATH = MODEL_DIR / "character_model.pth"


# ==========================================
# DEVICE
# ==========================================

if torch.cuda.is_available():

    device = torch.device("cuda")

    print(
        "Using GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    device = torch.device("cpu")

    print("Using CPU")


# ==========================================
# TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5,),
        (0.5,)
    )

])


# ==========================================
# LOAD EMNIST
# ==========================================

print("Loading EMNIST...")


dataset = datasets.EMNIST(

    root=str(DATASET_DIR),

    split="balanced",

    train=True,

    download=True,

    transform=transform

)


# ==========================================
# SELECT DIGITS + CAPITAL LETTERS
# ==========================================

target_classes = []


for index, class_name in enumerate(
    dataset.classes
):

    if class_name.isdigit():

        target_classes.append(index)

    elif (
        class_name.isalpha()
        and len(class_name) == 1
        and class_name.isupper()
    ):

        target_classes.append(index)


target_classes.sort()


# ==========================================
# CREATE LABEL MAPPING
# ==========================================

label_mapping = {

    original_label: new_label

    for new_label, original_label
    in enumerate(target_classes)

}


# ==========================================
# FIND TARGET SAMPLES
# ==========================================

target_indices = []


for index in range(len(dataset)):

    _, label = dataset[index]

    if label in label_mapping:

        target_indices.append(index)


print(
    f"Total target samples: "
    f"{len(target_indices)}"
)

print(
    f"Total classes: "
    f"{len(target_classes)}"
)


# ==========================================
# CUSTOM DATASET
# ==========================================

class CharacterDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        dataset,
        indices,
        label_mapping
    ):

        self.dataset = dataset

        self.indices = indices

        self.label_mapping = label_mapping


    def __len__(self):

        return len(self.indices)


    def __getitem__(self, index):

        real_index = self.indices[index]

        image, original_label = (
            self.dataset[real_index]
        )

        new_label = self.label_mapping[
            original_label
        ]

        return image, new_label


character_dataset = CharacterDataset(

    dataset,

    target_indices,

    label_mapping

)


# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

dataset_size = len(
    character_dataset
)


train_size = int(
    dataset_size * 0.9
)


validation_size = (
    dataset_size - train_size
)


train_dataset, validation_dataset = (
    torch.utils.data.random_split(

        character_dataset,

        [
            train_size,
            validation_size
        ],

        generator=torch.Generator()
        .manual_seed(42)

    )
)


print(
    f"Training samples: "
    f"{len(train_dataset)}"
)


print(
    f"Validation samples: "
    f"{len(validation_dataset)}"
)


# ==========================================
# DATA LOADERS
# ==========================================

BATCH_SIZE = 128


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ==========================================
# CNN MODEL
# ==========================================

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


# ==========================================
# CREATE MODEL
# ==========================================

model = CharacterCNN().to(device)


print()
print(model)


# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(

    model.parameters(),

    lr=0.001

)


# ==========================================
# TRAINING SETTINGS
# ==========================================

EPOCHS = 5


# ==========================================
# TRAINING LOOP
# ==========================================

for epoch in range(EPOCHS):

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )


    train_accuracy = (
        100.0 * correct / total
    )


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()


    validation_correct = 0

    validation_total = 0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            validation_total += (
                labels.size(0)
            )


            validation_correct += (
                predicted == labels
            ).sum().item()


    validation_accuracy = (

        100.0
        * validation_correct
        / validation_total

    )


    print(

        f"Epoch "
        f"{epoch + 1}/{EPOCHS} | "

        f"Loss: "
        f"{train_loss:.4f} | "

        f"Train Accuracy: "
        f"{train_accuracy:.2f}% | "

        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"

    )


# ==========================================
# SAVE MODEL
# ==========================================

torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "classes": [

            dataset.classes[index]

            for index in target_classes

        ]

    },

    MODEL_PATH

)


print()

print("=" * 50)

print("Training complete.")

print()

print(
    f"Model saved to:"
)

print(MODEL_PATH)

print("=" * 50)