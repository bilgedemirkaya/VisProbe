"""
RQ2: Adaptive vs Grid vs dense_grid on CIFAR-10 (ResNet-56)
Gaussian Noise + LabelConstant

- CIFAR-only (32x32), no ImageNet RN50.
- Identity test is lightweight (no analyses).
- Grid/Oracle use noise_sweep; Adaptive uses search(max_queries=BUDGET).

Run (Colab GPU example):
  %pip -q install --upgrade --force-reinstall torch==2.6.0+cu124 \
      torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
      --index-url https://download.pytorch.org/whl/cu124
  # restart runtime
  import os
  os.environ["VISPROBE_DEVICE"]="cuda:0"
  os.environ["RQ2_N"]="512"
  os.environ["RQ2_BUDGET"]="16"
  os.environ["RQ2_GRID_LEVELS"]="16"
  os.environ["RQ2_ORACLE_LEVELS"]="101"
  os.environ["RQ2_HI"]="0.50"
  os.environ["RQ2_SEED"]="1337"
  !visprobe run test_rq2_gaussian.py --device cuda:0
"""

import os
import random
from typing import List, Tuple

import torch
import torchvision.datasets as dsets
import torchvision.transforms as T

from visprobe.api.decorators import data_source, given, model, search
from visprobe.properties.classification import LabelConstant

# Your patched strategy that does: denorm → add σ·N(0,1) → clip → renorm; σ=0 is exact identity.
from visprobe.strategies.image import GaussianNoiseStrategy

# ----------------- Config -----------------
N_SAMPLES = int(os.getenv("RQ2_N", "256"))  # 256–512 on GPU is fine
LEVEL_HI = float(os.getenv("RQ2_HI", "0.50"))
ADAPTIVE_BUDGET = int(os.getenv("RQ2_BUDGET", "16"))  # equal to grid levels for fairness
GRID_LEVELS = int(os.getenv("RQ2_GRID_LEVELS", str(ADAPTIVE_BUDGET)))
DENSE_GRID_LEVELS = int(os.getenv("RQ2_DENSE_GRID_LEVELS", "101"))
SEED = int(os.getenv("RQ2_SEED", "1337"))
DEBUG = os.getenv("VF_DEBUG", "1") == "0"

random.seed(SEED)
torch.manual_seed(SEED)

# CIFAR-10 normalization
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

# ----------------- Banner -----------------


# ----------------- Data -----------------
def load_cifar10_as_tensor_label_lists(n: int) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    ds = dsets.CIFAR10(
        root=os.path.expanduser("~/.data"), train=False, download=True, transform=tfm
    )
    k = min(n, len(ds))
    images = [ds[i][0] for i in range(k)]
    labels = [int(ds[i][1]) for i in range(k)]
    return images, labels, ds.classes  # 10 labels


IMAGES, LABELS, CLASS_NAMES = load_cifar10_as_tensor_label_lists(N_SAMPLES)


def collate_stack(batch_list: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch_list, dim=0)


# ----------------- Model: CIFAR-10 ResNet-56 (Torch Hub) -----------------
# trust_repo=True avoids an interactive prompt on Colab
rn56 = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, trust_repo=True
).eval()


# Sanity: check labels vs predictions; optionally filter to clean-correct
def _pick_device() -> torch.device:
    env_device = os.environ.get("VISPROBE_DEVICE", "auto").lower()
    if env_device != "auto":
        return torch.device(env_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sanity_and_filter(
    model: torch.nn.Module, images: List[torch.Tensor], labels: List[int], class_names: List[str]
) -> List[torch.Tensor]:
    device = _pick_device()
    try:
        model = model.to(device)
    except Exception:
        pass
    batch = torch.stack(images, 0).to(device)
    with torch.no_grad():
        logits = model(batch)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
    pred = pred.detach().cpu().tolist()
    conf = conf.detach().cpu().tolist()

    total = len(images)
    correct_mask = [int(p) == int(y) for p, y in zip(pred, labels)]
    num_correct = sum(1 for v in correct_mask if v)
    if DEBUG:
        accuracy = num_correct / float(max(1, total))
        print(
            f"[SANITY] Clean accuracy on initial batch: "
            f"{num_correct}/{total} = {accuracy:.3f}"
        )
        for i in range(min(8, total)):
            y = labels[i]
            p = pred[i]
            gt = class_names[y]
            pred_name = class_names[p]
            print(
                f"  idx={i:02d} gt={gt!s:<10} "
                f"pred={pred_name!s:<10} conf={conf[i]:.2f}"
            )

    require_clean = os.getenv("RQ2_REQUIRE_CLEAN_CORRECT", "1") == "1"
    min_keep = int(os.getenv("RQ2_MIN_KEEP", "16"))
    if require_clean:
        filtered = [img for img, ok in zip(images, correct_mask) if ok]
        if len(filtered) >= min_keep:
            if DEBUG:
                print(
                    f"[SANITY] Using only correctly classified images: "
                    f"{len(filtered)} kept (>= {min_keep})."
                )
            return filtered
        else:
            if DEBUG:
                print(
                    f"[SANITY] Too few clean-correct samples "
                    f"({len(filtered)} < {min_keep}); keeping original set."
                )
    return images


# Apply sanity filtering before registering tests
IMAGES = _sanity_and_filter(rn56, IMAGES, LABELS, CLASS_NAMES)

if DEBUG:
    samp = torch.stack(IMAGES[:32], 0)
    print(f"[DATA] Loaded {len(IMAGES)} CIFAR images; sample tensor {tuple(samp.shape)}")

# ----------------- Tests -----------------


# 0) δ=0 identity — no sweeps
@given(
    strategy=GaussianNoiseStrategy(std_dev=0.0, mean=CIFAR_MEAN, std=CIFAR_STD),
    noise_sweep=None,
    resolutions=None,
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_identity_zero_noise(o, p):
    if DEBUG:
        print("[TEST] identity: logits", tuple(o["output"].shape), "→", tuple(p["output"].shape))
    assert LabelConstant.evaluate(o, p)


# 1) GRID — coarse sweep (levels ≈ budget)
@given(
    strategy=GaussianNoiseStrategy(std_dev=0.0, mean=CIFAR_MEAN, std=CIFAR_STD),
    noise_sweep={"levels": max(2, GRID_LEVELS + 1), "min_level": 0.0, "max_level": LEVEL_HI},
    property_name=f"LabelConst under GaussianNoise (Grid L={GRID_LEVELS+1})",
    resolutions=None,
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_rq2_grid_cifar_noise(o, p):
    if DEBUG:
        print(
            "[TEST] grid: σ=0 body; the sweep is logged by the runner. logits",
            tuple(o["output"].shape),
        )
    assert LabelConstant.evaluate(o, p)


# 2)  dense sweep
@given(
    strategy=GaussianNoiseStrategy(std_dev=0.0, mean=CIFAR_MEAN, std=CIFAR_STD),
    noise_sweep={"levels": max(2, DENSE_GRID_LEVELS), "min_level": 0.0, "max_level": LEVEL_HI},
    property_name=f"LabelConst under GaussianNoise (Oracle L={DENSE_GRID_LEVELS})",
    resolutions=None,
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_rq2_dense_grid_cifar_noise(o, p):
    if DEBUG:
        print(
            "[TEST] oracle: σ=0 body; dense sweep is logged by the runner. logits",
            tuple(o["output"].shape),
        )
    assert LabelConstant.evaluate(o, p)


def bounded_gaussian(level: float):
    # keep the search in a small, relevant window
    level = max(0.0, min(float(level), 0.02))
    return GaussianNoiseStrategy(std_dev=level, mean=CIFAR_MEAN, std=CIFAR_STD)


@search(
    strategy=bounded_gaussian,
    initial_level=0.0,
    step=0.01,  # first jump to 0.01 (likely fail under reducer="all")
    min_step=0.0025,  # ≤ 0.5 * dense step (0.005) → guarantees testing 0.005
    max_queries=14,  # enough to reach the 0.0025 tolerance cleanly
    resolutions=None,  # disable auxiliaries to keep query counts clean
    noise_sweep=None,
    top_k=None,
    reduce="all",
    property_name="LabelConst under GaussianNoise (Adaptive, tight)",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_rq2_adaptive_cifar_noise(o, p):
    if DEBUG:
        print("[TEST] adaptive: logits", tuple(o["output"].shape), "→", tuple(p["output"].shape))
    assert LabelConstant.evaluate(o, p)


# ----------------- Direct run (works even if the CLI doesn't auto-discover tests) -----------------
if __name__ == "__main__":
    # Allow running this file directly (e.g., via `visprobe run test_rq2_gaussian.py`).
    # Explicitly invoke each test wrapper so the runner executes and emits reports.
    # test_identity_zero_noise()
    # test_rq2_grid_cifar_noise()
    # test_rq2_dense_grid_cifar_noise()
    test_rq2_adaptive_cifar_noise()
