"""
RQ3 (VisProbe version): Property thresholds vs robustness correlations on CIFAR-10

Replicates test_rq3.py using VisProbe decorators, same dataset and model as RQ2 tests:
- Model: CIFAR-10 ResNet-56 from chenyaofo/pytorch-cifar-models
- Data: CIFAR-10 test set, normalized with CIFAR stats

We implement four fixed tests capturing per-sample metrics for:
- Gaussian noise (sigma)
- Brightness (factor)
- Gaussian blur (sigma)
- JPEG compression (severity)

These store: per-sample confidence drop and top-k overlap; the CSV/summary
correlation workflow in test_rq3.py can be reproduced downstream from the JSONs.
"""

import os
import random
import sys
from typing import List, Tuple

import torch
import torchvision.datasets as dsets
import torchvision.transforms as T

from visprobe.api.decorators import data_source, given, model, search
from visprobe.properties.classification import LabelConstant
from visprobe.strategies import (
    BrightnessStrategy,
    GaussianBlurStrategy,
    GaussianNoiseStrategy,
    JPEGCompressionStrategy,
)

# ----------------- Config -----------------
N_SAMPLES = int(os.getenv("RQ3_N", os.getenv("RQ2_N", "256")))
SEED = int(os.getenv("RQ3_SEED", os.getenv("RQ2_SEED", "1337")))
DEVICE_ENV = os.getenv("VISPROBE_DEVICE", "auto")
DEBUG = os.getenv("VF_DEBUG", "1") == "0"

random.seed(SEED)
torch.manual_seed(SEED)

# CIFAR-10 normalization
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]


# ----------------- Data -----------------
def load_cifar10_as_tensor_list(n: int) -> Tuple[List[torch.Tensor], List[str]]:
    tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    ds = dsets.CIFAR10(
        root=os.path.expanduser("~/.data"), train=False, download=True, transform=tfm
    )
    k = min(n, len(ds))
    images = [ds[i][0] for i in range(k)]
    return images, ds.classes


IMAGES, CLASS_NAMES = load_cifar10_as_tensor_list(N_SAMPLES)


def collate_stack(batch_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch_list, dim=0)


# ----------------- Model -----------------
rn56 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, trust_repo=True
).eval()


# ----------------- Tests -----------------

NOISE_MAX = float(os.getenv("RQ3_NOISE_MAX", "0.5"))
BRIGHT_MAX = float(os.getenv("RQ3_BRIGHTNESS_MAX", "0.8"))
BLUR_MAX = float(os.getenv("RQ3_BLUR_MAX", "3.0"))
JPEG_MAX = float(os.getenv("RQ3_JPEG_SEVERITY_MAX", "95"))


# Gaussian noise threshold via adaptive search
@search(
    strategy=lambda level: GaussianNoiseStrategy(
        std_dev=float(level), mean=CIFAR_MEAN, std=CIFAR_STD
    ),
    initial_level=0.0,
    step=max(NOISE_MAX / 4.0, 0.02),
    min_step=max(NOISE_MAX, 1e-6) / (2**7),
    max_queries=48,
    resolutions=None,
    noise_sweep=None,
    top_k=5,
    reduce="all",
    property_name="LabelConst under GaussianNoise (Adaptive)",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_rq3_gaussian_noise(o, p):
    assert LabelConstant.evaluate(o, p)


# Brightness (lighten) threshold
@search(
    strategy=lambda level: BrightnessStrategy(brightness_factor=1.0 + float(level)),
    initial_level=0.0,
    step=max(BRIGHT_MAX / 4.0, 0.02),
    min_step=max(BRIGHT_MAX, 1e-6) / (2**7),
    max_queries=48,
    resolutions=None,
    noise_sweep=None,
    top_k=5,
    reduce="all",
    property_name="LabelConst under Brightness-Lighten (Adaptive)",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_rq3_brightness_lighten(o, p):
    assert LabelConstant.evaluate(o, p)


# Brightness (darken) threshold
@search(
    strategy=lambda level: BrightnessStrategy(brightness_factor=max(0.0, 1.0 - float(level))),
    initial_level=0.0,
    step=max(BRIGHT_MAX / 4.0, 0.02),
    min_step=max(BRIGHT_MAX, 1e-6) / (2**7),
    max_queries=48,
    resolutions=None,
    noise_sweep=None,
    top_k=5,
    reduce="all",
    property_name="LabelConst under Brightness-Darken (Adaptive)",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_rq3_brightness_darken(o, p):
    assert LabelConstant.evaluate(o, p)


@search(
    strategy=lambda level: GaussianBlurStrategy(sigma=float(level)),
    initial_level=0.0,
    step=max(BLUR_MAX / 4.0, 0.05),
    min_step=max(BLUR_MAX, 1e-6) / (2**7),
    max_queries=48,
    resolutions=None,
    noise_sweep=None,
    top_k=5,
    reduce="all",
    property_name="LabelConst under GaussianBlur (Adaptive)",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_rq3_blur(o, p):
    assert LabelConstant.evaluate(o, p)


@search(
    strategy=lambda level: JPEGCompressionStrategy(severity=float(level)),
    initial_level=0.0,
    step=max(JPEG_MAX / 8.0, 2.0),
    min_step=max(JPEG_MAX, 1e-6) / (2**7),
    max_queries=56,
    resolutions=None,
    noise_sweep=None,
    top_k=5,
    reduce="all",
    property_name="LabelConst under JPEGCompression (Adaptive)",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_rq3_jpeg(o, p):
    assert LabelConstant.evaluate(o, p)


if __name__ == "__main__":
    print("[RUN] RQ3 VisProbe tests…")
    test_rq3_gaussian_noise()
    test_rq3_brightness_lighten()
    test_rq3_brightness_darken()
    test_rq3_blur()
    test_rq3_jpeg()
    print("[RUN] done. JSONs in /tmp/visprobe_results")
