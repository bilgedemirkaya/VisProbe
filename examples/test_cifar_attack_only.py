"""
CIFAR-10 attack-only baseline: FGSM(8/255), strict top-1 equality.
Matches dataset/model/normalization used in test_rq2_gaussian.py.
"""

import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as T

from visprobe.api.decorators import data_source, given, model
from visprobe.properties.classification import LabelConstant

# Device and threading to avoid OOM
os.environ["VISPROBE_DEVICE"] = "cpu"
try:
    torch.set_num_threads(max(1, int(os.getenv("VF_THREADS", "1"))))
except Exception:
    pass
torch.manual_seed(int(os.getenv("RQ2_SEED", "1337")))


# CIFAR-10 normalization (same as test_rq2_gaussian.py)
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


# Model-space FGSM (avoid ART and device issues)
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


# Data and model
N_SAMPLES = int(os.getenv("RQ2_N", "1"))
IMAGES, CLASS_NAMES = load_cifar10_as_tensor_list(N_SAMPLES, CIFAR_MEAN, CIFAR_STD)
rn56 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, trust_repo=True
).eval()


@given(
    strategy=[fgsm_model_space(eps=8 / 255)],
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
    assert LabelConstant.evaluate(original, perturbed)


if __name__ == "__main__":
    test_attack_only_fgsm_top1()
    print("Attack-only run complete. JSON in /tmp/visprobe_results")
