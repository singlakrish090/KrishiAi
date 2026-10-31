import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ai_model" / "wheat_disease_classifier.pth"
CLASS_NAMES_PATH = PROJECT_ROOT / "ai_model" / "wheat_class_names.json"

DISPLAY_NAMES = {
    "wheat_aphid": "Aphid Pest Damage",
    "wheat_blast": "Wheat Blast",
    "wheat_healthy": "Healthy Wheat",
    "wheat_leaf_rust": "Wheat Leaf Rust",
    "wheat_smut": "Wheat Smut",
    "wheat_yellow_rust": "Wheat Yellow Rust",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        len(class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


MODEL, CLASS_NAMES = load_model()


def predict_wheat_health(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = TRANSFORM(image).unsqueeze(0)

    with torch.inference_mode():
        logits = MODEL(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = CLASS_NAMES[predicted_index]

    return {
        "predicted_class": predicted_class,
        "label": DISPLAY_NAMES.get(predicted_class, predicted_class),
        "confidence": round(
            float(probabilities[predicted_index].item()) * 100, 2
        ),
        "probabilities": {
            DISPLAY_NAMES.get(name, name): round(
                float(probabilities[index].item()) * 100, 2
            )
            for index, name in enumerate(CLASS_NAMES)
        },
        "disclaimer": (
            "This is an AI prototype prediction, not a confirmed "
            "agricultural diagnosis. Please verify suspected crop "
            "problems with an agricultural expert."
        ),
    }