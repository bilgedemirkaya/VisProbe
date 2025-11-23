"""
This is a test that uses a torch.hub.load model as the model.
"""

import torch
from cifar10_models.vgg import vgg11_bn
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from visprobe.api.decorators import data_source, model, search
from visprobe.api.utils import cifar10_data_source
from visprobe.properties.classification import LabelConstant
from visprobe.strategies import FGSMStrategy

# Pretrained model
my_model = vgg11_bn(pretrained=True)
my_model.eval()


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


@search(strategy=FGSMStrategy(eps=0.001), initial_level=0.01, step=0.01)
@model(my_model)
@data_source(data_obj=data_obj, collate_fn=collate_fn, class_names=class_names, mean=mean, std=std)
def test_fgsm_on_cats(original, perturbed):
    """
    This test searches for the minimal FGSM perturbation that causes a
    misclassification on cat images.
    """
    assert LabelConstant.evaluate(original, perturbed)


def main():
    test_fgsm_on_cats()


if __name__ == "__main__":
    main()
