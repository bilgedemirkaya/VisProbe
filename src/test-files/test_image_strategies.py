"""
This module contains tests for various image transformation strategies,
such as brightness and rotation, to ensure the model's predictions
remain robust.
"""

import requests
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from visprobe.api.decorators import data_source, given, model
from visprobe.strategies import BrightnessStrategy, RotateStrategy

# --- Configuration ---
try:
    imagenet_labels_url = (
        "https://raw.githubusercontent.com/anishathalye/"
        "imagenet-simple-labels/master/imagenet-simple-labels.json"
    )
    response = requests.get(imagenet_labels_url)
    response.raise_for_status()
    IMAGENET_LABELS = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error loading ImageNet labels: {e}")
    IMAGENET_LABELS = [str(i) for i in range(1000)]

weights = ResNet50_Weights.IMAGENET1K_V1
model_instance = resnet50(weights=weights)
preprocess = weights.transforms()


# --- Test Helpers ---
def load_image(url: str) -> torch.Tensor:
    """Loads and preprocesses an image from a URL."""
    try:
        response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        return preprocess(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return torch.randn(1, 3, 224, 224)


def get_top_prediction(preds_batch):
    """Gets the top prediction from a batch of predictions."""
    predictions = preds_batch[0]
    pred_idx = torch.argmax(predictions, dim=0)
    return weights.meta["categories"][pred_idx.item()]


# --- VisProbe Tests ---
@given(strategy=BrightnessStrategy(brightness_factor=1.5))
@model(model_instance)
@data_source(
    data_obj=[load_image("https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg")],
    collate_fn=torch.vstack,
    class_names=IMAGENET_LABELS,
    mean=preprocess.mean,
    std=preprocess.std,
)
def test_brightness_robustness(original, perturbed):
    """Tests robustness against a brightness increase."""
    original_preds, _ = original["output"]
    perturbed_preds, _ = perturbed["output"]

    orig_label = get_top_prediction(original_preds)
    pert_label = get_top_prediction(perturbed_preds)

    assert orig_label == pert_label


@given(strategy=RotateStrategy(angle=30))
@model(model_instance)
@data_source(
    data_obj=[load_image("https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg")],
    collate_fn=torch.vstack,
    class_names=IMAGENET_LABELS,
    mean=preprocess.mean,
    std=preprocess.std,
)
def test_rotation_robustness(original, perturbed):
    """Tests robustness against a 30-degree rotation."""
    original_preds, _ = original["output"]
    perturbed_preds, _ = perturbed["output"]

    orig_label = get_top_prediction(original_preds)
    pert_label = get_top_prediction(perturbed_preds)

    assert orig_label == pert_label


# --- Main Execution ---
if __name__ == "__main__":
    print("🔬 Running brightness test...")
    brightness_result = test_brightness_robustness()
    passed = brightness_result.passed_samples
    total = brightness_result.total_samples
    print(f"✅ Test complete. Passed samples: {passed}/{total}")

    print("\n🔬 Running rotation test...")
    rotation_result = test_rotation_robustness()
    passed = rotation_result.passed_samples
    total = rotation_result.total_samples
    print(f"✅ Test complete. Passed samples: {passed}/{total}")

    print(
        f"\n💡 Tip: run `visprobe visualize {__file__}` "
        "for the interactive dashboard."
    )
