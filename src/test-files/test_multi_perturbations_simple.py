"""
Example tests demonstrating multiple perturbations with auto device management.

This version uses visprobe.auto_init to automatically handle device management,
threading, and other stability configurations.
"""

from __future__ import annotations

import torch
from cifar10_models.vgg import vgg11_bn
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

# Auto-configure VisProbe for stability (device, threading, warnings)
import visprobe.auto_init
from visprobe.api.decorators import data_source, given, model, search
from visprobe.api.utils import cifar10_data_source
from visprobe.properties.classification import LabelConstant
from visprobe.strategies import FGSMStrategy

# --- Model (no manual device management needed) ---
my_model = vgg11_bn(pretrained=True)
my_model.eval()


# --- Data (CIFAR-10 cats subset, normalized for the model) ---
def build_cat_subset_normalized(num_images: int = 16):
    tfm = Compose(
        [ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])]
    )
    ds = CIFAR10(root="./data", train=False, download=True, transform=tfm)
    # Filter to cats (label 3)
    cat_indices = [i for i, lbl in enumerate(ds.targets) if lbl == 3]
    subset = torch.utils.data.Subset(ds, cat_indices[:num_images])
    return cifar10_data_source(
        subset, normalized=True, meta_path="./data/cifar-10-batches-py/batches.meta"
    )


data_obj, collate_fn, class_names, mean, std = build_cat_subset_normalized(num_images=16)

# --- Multiple perturbations ---
multi_given = [
    {"type": "gaussian_noise", "std_dev": 0.02},
    {"type": "brightness", "brightness_factor": 1.05},
]

multi_search = [
    {"type": "gaussian_noise", "std_dev": 0.001},
    FGSMStrategy(eps=0.001),
]


@given(strategy=multi_given, property_name="Label Consistency (multi, auto-init)")
@model(my_model)
@data_source(data_obj=data_obj, collate_fn=collate_fn, class_names=class_names, mean=mean, std=std)
def test_given_multi(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)


@search(
    strategy=multi_search,
    initial_level=0.002,
    step=0.002,
    property_name="Label Const (search, multi, auto-init)",
)
@model(my_model)
@data_source(data_obj=data_obj, collate_fn=collate_fn, class_names=class_names, mean=mean, std=std)
def test_search_multi(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)


def main():
    test_given_multi()
    # Skip search test for quick demo
    # test_search_multi()


if __name__ == "__main__":
    main()
