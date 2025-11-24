"""
This is an end-to-end test that uses VisProbe to find the minimum
perturbation required to fool a ResNet50 model on a cat image.
"""

import requests
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from visprobe.api.decorators import data_source, model, search
from visprobe.strategies import FGSMStrategy

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

ENSEMBLE_LAYERS = ["layer1", "layer2", "layer3", "layer4", "fc"]
weights = ResNet50_Weights.IMAGENET1K_V1
model_instance = resnet50(weights=weights)
preprocess = weights.transforms()


# --- Test Helpers ---
def load_image(url: str, size: tuple = (224, 224)) -> torch.Tensor:
    """Loads and preprocesses an image from a URL."""
    try:
        response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        return preprocess(image).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return torch.randn(1, 3, *size)


def get_top_prediction(preds_batch):
    """Gets the top prediction and confidence from a batch of predictions."""
    predictions = preds_batch[0]
    probabilities = torch.nn.functional.softmax(predictions, dim=0)
    confidence, pred_idx = torch.max(probabilities, 0)
    return IMAGENET_LABELS[pred_idx.item()], confidence.item()


# --- VisProbe Test ---
@search(
    strategy=lambda eps: FGSMStrategy(eps=eps),
    initial_level=0.001,
    step=0.001,
    min_step=0.00001,
    max_queries=500,
)
@model(model_instance, capture_intermediate_layers=ENSEMBLE_LAYERS)
@data_source(
    data_obj=[load_image("https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg")],
    collate_fn=torch.vstack,
    class_names=IMAGENET_LABELS,
    mean=preprocess.mean,
    std=preprocess.std,
)
def test_cat_attack(original, perturbed):
    """
    Asserts that the model's prediction for the original and perturbed
    images remains the same.
    """
    original_preds, _ = original["output"]
    perturbed_preds, _ = perturbed["output"]

    orig_label, _ = get_top_prediction(original_preds)
    pert_label, _ = get_top_prediction(perturbed_preds)

    print(f"Original prediction: {orig_label}, Perturbed prediction: {pert_label}")

    assert orig_label == pert_label


# --- Main Execution ---
if __name__ == "__main__":
    result = test_cat_attack()

    if result.failure_threshold is not None:
        print("\n--- 💡 VisProbe Search Complete ---")
        print(f"Failure Threshold: ε = {result.failure_threshold:.4f}")
        print(f"Model Queries: {result.model_queries}")
    else:
        print("\n--- ✅ VisProbe Search Complete ---")
        print("No failure point found within the given search parameters.")

    print(f"\n💡 Tip: run   visprobe visualize {__file__}   for the dashboard.")
