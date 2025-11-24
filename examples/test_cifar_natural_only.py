"""
CIFAR-10 natural-only baseline: Gaussian noise (σ=0.05), strict top-1 equality.
Matches dataset/model/normalization used in test_rq2_gaussian.py.
"""

import os
from typing import List, Tuple

import torch
import torchvision.datasets as dsets
import torchvision.transforms as T

from visprobe.api.decorators import data_source, given, model
from visprobe.properties.classification import LabelConstant
from visprobe.strategies.image import GaussianNoiseStrategy

os.environ["VISPROBE_DEVICE"] = "cpu"
try:
    torch.set_num_threads(max(1, int(os.getenv("VF_THREADS", "1"))))
except Exception:
    pass
torch.manual_seed(int(os.getenv("RQ2_SEED", "1337")))

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]


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


def collate_stack(batch_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch_list, dim=0)


N_SAMPLES = int(os.getenv("RQ2_N", "1"))
IMAGES, CLASS_NAMES = load_cifar10_as_tensor_list(N_SAMPLES, CIFAR_MEAN, CIFAR_STD)
rn56 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, trust_repo=True
).eval()


@given(
    strategy=[GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD)],
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
def test_natural_only_top1(original, perturbed):
    assert LabelConstant.evaluate(original, perturbed)


if __name__ == "__main__":
    test_natural_only_top1()
    print("Natural-only run complete. JSON in /tmp/visprobe_results")
