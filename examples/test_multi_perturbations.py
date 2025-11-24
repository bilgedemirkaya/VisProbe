"""
Example tests demonstrating multiple perturbations passed as an array.

Both @given and @search accept a single perturbation or a list/tuple;
when a list/tuple is provided, the perturbations are applied sequentially.
"""

from __future__ import annotations

import os

import torch
from cifar10_models.vgg import vgg11_bn
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from visprobe.api.decorators import data_source, given, model, search
from visprobe.api.utils import cifar10_data_source
from visprobe.properties.classification import LabelConstant
from visprobe.strategies import FGSMStrategy

# --- Device Management ---
# Force CPU to avoid MPS/CUDA device mismatches
os.environ["VISPROBE_DEVICE"] = "cpu"
torch.set_num_threads(max(1, int(os.getenv("VF_THREADS", "1"))))

# --- Model ---
my_model = vgg11_bn(pretrained=True)
my_model.eval()
# Explicitly move to CPU to prevent device mismatch
my_model = my_model.cpu()


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
# These will execute in order: Gaussian noise, then a small brightness shift
multi_given = [
    {"type": "gaussian_noise", "std_dev": 0.02},
    {"type": "brightness", "brightness_factor": 1.05},
]

# For search, demonstrate composition of a level-driven image-space noise and a gradient-based FGSM
multi_search = [
    {
        "type": "gaussian_noise",
        "std_dev": 0.001,
    },  # std will be overridden by the current search level
    FGSMStrategy(eps=0.001),  # eps will be overridden by the current search level
]


@given(strategy=multi_given, property_name="Label Consistency")
@model(my_model)
@data_source(data_obj=data_obj, collate_fn=collate_fn, class_names=class_names, mean=mean, std=std)
def test_given_multi(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)


@search(
    strategy=multi_search,
    initial_level=0.002,
    step=0.002,
    property_name="Label Const (search, multi)",
)
@model(my_model)
@data_source(data_obj=data_obj, collate_fn=collate_fn, class_names=class_names, mean=mean, std=std)
def test_search_multi(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)


def main():
    test_given_multi()
    test_search_multi()


if __name__ == "__main__":
    main()
