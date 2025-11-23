"""
CIFAR-10 mixed scenario: Gaussian noise (σ=0.05) + FGSM(8/255) with
SafetyPreservation property (top-1 contained in top-5 + confidence drop ≤ 0.5).
"""

import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as T

from visprobe.api.decorators import data_source, given, model
from visprobe.properties import ConfidenceDrop, Property, TopKStability
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


def fgsm_model_space(eps: float):
    def _fn(level: float | None, imgs: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        step = eps if level is None else float(level)
        imgs_adv = imgs.detach().clone()
        imgs_adv.requires_grad_(True)
        model.eval()
        logits = model(imgs_adv)
        targets = torch.argmax(logits.detach(), dim=1)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        grad_sign = imgs_adv.grad.detach().sign()
        adv = imgs_adv + step * grad_sign
        return adv.detach()

    return _fn


class SafetyPreservation(Property):
    def __init__(self, k: int = 5, max_drop: float = 0.5):
        self.topk = TopKStability(k=k, mode="containment", require_containment=True)
        self.conf = ConfidenceDrop(max_drop=max_drop)

    def __call__(self, original, perturbed) -> bool:
        return self.topk(original, perturbed) and self.conf(original, perturbed)


N_SAMPLES = int(os.getenv("RQ2_N", "1"))
IMAGES, CLASS_NAMES = load_cifar10_as_tensor_list(N_SAMPLES, CIFAR_MEAN, CIFAR_STD)
rn56 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, trust_repo=True
).eval()


@given(
    strategy=[
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        fgsm_model_space(eps=8 / 255),
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
def test_cifar_mixed_scenario(original, perturbed):
    assert SafetyPreservation(k=5, max_drop=0.5)(original, perturbed)


if __name__ == "__main__":
    test_cifar_mixed_scenario()
    print("Scenario run complete. JSON in /tmp/visprobe_results")
