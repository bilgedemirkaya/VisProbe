import json
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as T

# Auto-configure VisProbe for stability
import visprobe.auto_init
from visprobe.api.decorators import data_source, given, model
from visprobe.properties import ConfidenceDrop, LabelConstant, Property, TopKStability
from visprobe.strategies import BrightnessStrategy, FGSMStrategy, GaussianNoiseStrategy

# ----------------- Reproducibility -----------------
torch.manual_seed(0)


# ----------------- CIFAR-10 data helpers -----------------
def load_cifar10_as_tensor_list(
    n: int, mean: List[float], std: List[float]
) -> Tuple[List[torch.Tensor], List[str]]:
    tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds = dsets.CIFAR10(
        root=os.path.expanduser("~/.data"), train=False, download=True, transform=tfm
    )
    k = min(n, len(ds))
    images = [ds[i][0] for i in range(k)]
    return images, ds.classes


# ----------------- Model & preprocessing -----------------

# CIFAR-10 normalization
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

# Data
N_SAMPLES = int(os.getenv("RQ2_N", "32"))
IMAGES, CLASS_NAMES = load_cifar10_as_tensor_list(N_SAMPLES, CIFAR_MEAN, CIFAR_STD)


def collate_stack(batch_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch_list, dim=0)


# Model: CIFAR-10 ResNet-56 (pretrained)
rn56 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, trust_repo=True
).eval()


# ----------------- Properties -----------------
class OverconfidentWrongPrediction(Property):
    """
    Detects concerning cases where:
    - Label flips (prediction changes)
    - BUT confidence doesn't drop significantly (model stays confident about wrong answer)
    """

    def __init__(self, max_drop: float = 0.1):
        self.max_drop = max_drop

    def __call__(self, original, perturbed) -> bool:
        # Label flipped AND confidence maintained = concerning case
        label_flipped = not LabelConstant.evaluate(original, perturbed)
        confidence_maintained = not ConfidenceDrop.evaluate(
            original, perturbed, max_drop=self.max_drop
        )
        return label_flipped and confidence_maintained

    def __str__(self) -> str:
        return f"OverconfidentWrongPrediction(max_drop={self.max_drop})"


# ----------------- Properties -----------------
class SafetyPreservation(Property):
    """
    Safety-relevant stability under mixed corruptions:
    - Containment: original top-1 must appear in perturbed top-k.
    - Confidence guardrail: drop in top-1 confidence <= max_drop.
    """

    def __init__(self, k: int = 5, max_drop: float = 0.5):
        self.topk = TopKStability(k=k, mode="containment", require_containment=True)
        self.conf = ConfidenceDrop(max_drop=max_drop)

    def __call__(self, original, perturbed) -> bool:
        return self.topk(original, perturbed) and self.conf(original, perturbed)

    def __str__(self) -> str:
        return f"SafetyPreservation(topk=5 containment, max_drop={self.conf.max_drop})"


# ----------------- Tests (CIFAR-10) -----------------
# Baseline A: attack-only FGSM on CIFAR-10, strict top-1 equality
@given(
    strategy=[FGSMStrategy(eps=8 / 255)],
    property_name="CIFAR10: Attack-only FGSM(8/255) (model-space), top-1 equality",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_attack_only_fgsm_top1(original, perturbed):
    # Check for overconfident wrong predictions (concerning case)
    overconfident_wrong = OverconfidentWrongPrediction.evaluate(original, perturbed)
    if overconfident_wrong:
        print("🚨 CONCERNING: Model flipped label but maintained confidence!")

    # Original assertion - should maintain label
    assert LabelConstant.evaluate(original, perturbed)


# Baseline B: natural-only Gaussian noise, strict top-1 equality
@given(
    strategy=[
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        BrightnessStrategy(brightness_factor=1.5),
    ],
    property_name="CIFAR10: Natural-only Gaussian noise(σ=0.05), top-1 equality",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_bad_weather(original, perturbed):
    # Check for overconfident wrong predictions (concerning case)
    overconfident_wrong = OverconfidentWrongPrediction.evaluate(original, perturbed)
    if overconfident_wrong:
        print("🚨 CONCERNING: Model flipped label but maintained confidence!")

    # Original assertion - confidence should drop under natural corruptions
    assert ConfidenceDrop.evaluate(original, perturbed)


# Scenario: natural corruption (Gaussian noise) + adversarial pressure (FGSM), safety-aware property
@given(
    strategy=[
        BrightnessStrategy(brightness_factor=0.85),
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        FGSMStrategy(eps=8 / 255),
    ],
    property_name="CIFAR10: Top-1-in-Top-5 + bounded confidence under noise+FGSM",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_mixed_real_world_autonomous_driving(original, perturbed):
    # Check for overconfident wrong predictions (concerning case)
    overconfident_wrong = OverconfidentWrongPrediction.evaluate(original, perturbed)
    if overconfident_wrong:
        print("🚨 CONCERNING: Model flipped label but maintained confidence!")

    # Original assertion - safety preservation under mixed corruptions
    assert SafetyPreservation(k=5, max_drop=0.5)(original, perturbed)


# Dedicated test to find overconfident wrong predictions
@given(
    strategy=[
        BrightnessStrategy(brightness_factor=0.7),
        GaussianNoiseStrategy(std_dev=0.08, mean=CIFAR_MEAN, std=CIFAR_STD),
        FGSMStrategy(eps=12 / 255),
    ],
    property_name="CIFAR10: Detect overconfident wrong predictions",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_overconfident_wrong_predictions(original, perturbed):
    # This test specifically looks for the concerning case
    assert OverconfidentWrongPrediction.evaluate(original, perturbed)


def test_real_world_autonomous_driving(mode: str):
    if mode == "attack":
        test_attack_only_fgsm_top1()
    elif mode == "natural":
        test_bad_weather()
    elif mode == "overconfident":
        test_overconfident_wrong_predictions()
    else:
        test_mixed_real_world_autonomous_driving()


if __name__ == "__main__":
    mode = (os.getenv("VF_WHICH", "mixed") or "mixed").lower().strip()
    print(f"🔍 Running {mode} test with overconfident wrong prediction detection...")
    test_real_world_autonomous_driving(mode)
