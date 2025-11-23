import os

# --- macOS-friendly runtime hygiene ---
import warnings
from typing import List, Tuple

import torch
import torchvision.datasets as dsets
import torchvision.transforms as T

from visprobe.api.decorators import data_source, given, model
from visprobe.properties import ConfidenceDrop, L2Distance, LabelConstant, Property, TopKStability
from visprobe.strategies import BrightnessStrategy, FGSMStrategy, GaussianNoiseStrategy

# Silence the torchvision antialias deprecation warning (benign but noisy)
warnings.filterwarnings("ignore", message=".*antialias.*will change from None to True.*")

# Force CPU and keep threads low (prevents memory spikes / oversubscription)
os.environ.setdefault("VISPROBE_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(max(1, int(os.getenv("VF_THREADS", "1"))))
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

# ----------------- Reproducibility -----------------
torch.manual_seed(0)
try:
    torch.set_num_threads(max(1, int(os.getenv("VF_THREADS", "1"))))
except Exception:
    pass


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
os.environ["VISPROBE_DEVICE"] = "cpu"

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


# A) ATTACK-ONLY (baseline): FGSM, strict top-1 equality
@given(strategy=[FGSMStrategy(eps=8 / 255)], property_name="CIFAR10: FGSM(8/255) — top-1 equality")
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_attack_only_fgsm_top1(o, p):
    assert LabelConstant.evaluate(o, p)


# A2) ATTACK-ONLY (complement): top-5 containment + confidence guardrail
@given(
    strategy=[FGSMStrategy(eps=8 / 255)],
    property_name="CIFAR10: FGSM(8/255) — top-5 contains clean top-1",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_attack_only_fgsm_top5(o, p):
    assert TopKStability.evaluate(o, p, k=5, mode="containment", require_containment=True)


@given(
    strategy=[FGSMStrategy(eps=8 / 255)],
    property_name="CIFAR10: FGSM(8/255) — confidence guardrail",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_attack_only_fgsm_conf(o, p):
    assert ConfidenceDrop.evaluate(o, p, max_drop=0.3)


# B) NATURAL-ONLY baseline: noise + brightness (no adversary)
@given(
    strategy=[
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        BrightnessStrategy(brightness_factor=1.5),
    ],
    property_name="CIFAR10: noise+brightness — top-1 equality",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_natural_only_top1(o, p):
    assert LabelConstant.evaluate(o, p)


@given(
    strategy=[
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        BrightnessStrategy(brightness_factor=1.5),
    ],
    property_name="CIFAR10: noise+brightness — confidence guardrail",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_natural_only_conf(o, p):
    assert ConfidenceDrop.evaluate(o, p, max_drop=0.3)


# C) MIXED SCENARIO: brightness -> noise -> FGSM (domain-specific)
@given(
    strategy=[
        BrightnessStrategy(brightness_factor=0.85),
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        FGSMStrategy(eps=8 / 255),
    ],
    property_name="CIFAR10: scenario — safety = top-5 containment + conf guard",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_scenario_safety(o, p):
    ok_top5 = TopKStability.evaluate(o, p, k=5, mode="containment", require_containment=True)
    ok_conf = ConfidenceDrop.evaluate(o, p, max_drop=0.5)
    assert ok_top5 and ok_conf


# C2) MIXED: exact top-k set unchanged (strict)
@given(
    strategy=[
        BrightnessStrategy(brightness_factor=0.85),
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        FGSMStrategy(eps=8 / 255),
    ],
    property_name="CIFAR10: scenario — top-k set identical (Jaccard=1.0)",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_scenario_topk_identical(o, p):
    assert TopKStability.evaluate(o, p, k=5, mode="jaccard", min_jaccard=1.0)


# C3) MIXED: rank ceiling — clean class must stay within top-3
@given(
    strategy=[
        BrightnessStrategy(brightness_factor=0.85),
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        FGSMStrategy(eps=8 / 255),
    ],
    property_name="CIFAR10: scenario — rank ceiling (clean ∈ top-3)",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_scenario_rank_top3(o, p):
    assert TopKStability.evaluate(o, p, k=3, mode="containment", require_containment=True)


# C4) MIXED: bounded logit drift
@given(
    strategy=[
        BrightnessStrategy(brightness_factor=0.85),
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        FGSMStrategy(eps=8 / 255),
    ],
    property_name="CIFAR10: scenario — logits L2 ≤ 1.0",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_scenario_logit_l2(o, p):
    assert L2Distance.evaluate(o, p, max_delta=1.0)


# C5) MIXED: no “silent degradation”
# (if label & top-k are unchanged, confidence must also be OK)
@given(
    strategy=[
        BrightnessStrategy(brightness_factor=0.85),
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        FGSMStrategy(eps=8 / 255),
    ],
    property_name="CIFAR10: scenario — no silent degradation",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_scenario_no_silent_degradation(o, p):
    same_label = LabelConstant.evaluate(o, p)
    topk_same = TopKStability.evaluate(o, p, k=5, mode="jaccard", min_jaccard=1.0)
    conf_ok = ConfidenceDrop.evaluate(o, p, max_drop=0.3)
    assert (not (same_label and topk_same)) or conf_ok


# C0) MIXED (strict): top-1 label must remain unchanged
@given(
    strategy=[
        BrightnessStrategy(brightness_factor=0.85),
        GaussianNoiseStrategy(std_dev=0.05, mean=CIFAR_MEAN, std=CIFAR_STD),
        FGSMStrategy(eps=8 / 255),
    ],
    property_name="CIFAR10: scenario — top-1 equality (strict)",
)
@model(rn56)
@data_source(
    data_obj=IMAGES,
    collate_fn=collate_stack,
    class_names=CLASS_NAMES,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
)
def test_scenario_top1_strict(o, p):
    assert LabelConstant.evaluate(o, p)


# --- add this below your test functions ---


def _run_safely(fn, name):
    try:
        rep = fn()  # many VisProbe runners return a report-like object
        print(f"[OK] {name}")
        return {"name": name, "ok": True, "report": rep}
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        return {"name": name, "ok": False, "report": None}
    except Exception as e:
        # keep going so one failure doesn't stop the rest
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")
        return {"name": name, "ok": False, "report": None}


def test_real_world_autonomous_driving(mode: str):
    """
    mode in {"attack","natural","mixed","all"} (default: "mixed")
    Runs the selected group of tests and prints a small summary.
    """
    mode = (mode or "mixed").strip().lower()
    results = []

    if mode in ("attack", "adv"):
        results.append(_run_safely(test_attack_only_fgsm_top1, "attack_only: top1"))
        results.append(_run_safely(test_attack_only_fgsm_top5, "attack_only: top5 containment"))
        results.append(_run_safely(test_attack_only_fgsm_conf, "attack_only: confidence guard"))
    elif mode in ("natural", "nat"):
        results.append(_run_safely(test_natural_only_top1, "natural_only: top1"))
        results.append(_run_safely(test_natural_only_conf, "natural_only: confidence guard"))
    elif mode in ("mixed", "scenario"):
        # results.append(
        #     _run_safely(test_scenario_safety, "scenario: safety (top5+conf)")
        # )
        # results.append(
        #     _run_safely(test_scenario_topk_identical, "scenario: topk identical (J=1.0)")
        # )
        # results.append(
        #     _run_safely(test_scenario_rank_top3, "scenario: rank ceiling (≤3)")
        # )
        # results.append(
        #     _run_safely(test_scenario_logit_l2, "scenario: logits L2 ≤ 1.0")
        # )
        # results.append(
        #     _run_safely(test_scenario_no_silent_degradation, "scenario: no silent degradation")
        # )
        results.append(_run_safely(test_scenario_top1_strict, "scenario: top-1 equality (strict)"))
    else:
        print(f"[warn] Unknown mode '{mode}', defaulting to 'mixed'.")
        return test_real_world_autonomous_driving("mixed")

    # tiny summary (pass/fail per test); full per-image stats live in VisProbe JSONs
    ok = sum(1 for r in results if r["ok"])
    tot = len(results)
    print(f"\n[summary] passed {ok}/{tot} checks (mode={mode}).")
    print("         Detailed per-image results are in /tmp/visprobe_results/*.json")
    return results


if __name__ == "__main__":
    mode = (os.getenv("VF_WHICH", "mixed") or "mixed").lower().strip()
    test_real_world_autonomous_driving(mode)
