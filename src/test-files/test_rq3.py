import argparse
import io
import json
import os
import random
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

# -------------------------
# Attempt to import Exp1 utilities
# Must provide:
#   get_model(device) -> torch.nn.Module in eval mode (weights loaded)
#   get_test_loader(batch_size, num_workers=2, shuffle=False) -> DataLoader over CIFAR-10 test set
# Optional:
#   get_normalize() -> torchvision.transforms.Normalize used in training
# -------------------------
EXP1_IMPORT_CANDIDATES = [
    "exp1_setup",
    "experiment1",
    "exp1_utils",
    "src.exp1_setup",
    "experiments.exp1",
]
exp1 = None
for name in EXP1_IMPORT_CANDIDATES:
    try:
        exp1 = __import__(name, fromlist=["*"])
        break
    except Exception:
        pass

if exp1 is None:

    class _MissingExp1(Exception):
        pass

    def get_model(device):
        raise _MissingExp1(
            "Experiment 1 utilities not found. Provide get_model(device) in "
            "exp1_setup.py (or adjust import)."
        )

    def get_test_loader(batch_size, num_workers=2, shuffle=False):
        raise _MissingExp1(
            "Provide get_test_loader(batch_size, num_workers, shuffle) for "
            "CIFAR-10 test set (shuffle=False)."
        )

    def get_normalize():
        # Default to CIFAR-10 normalization if Exp1 normalize isn't available.
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        return transforms.Normalize(mean, std)

else:
    get_model = getattr(exp1, "get_model")
    get_test_loader = getattr(exp1, "get_test_loader")
    get_normalize = getattr(
        exp1,
        "get_normalize",
        lambda: transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    )


# -------------------------
# Utilities
# -------------------------
def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_forward(model: nn.Module, normalizer: Callable, already_normalized: bool):
    """Returns a callable f(x) that applies normalization iff dataset isn't normalized yet."""
    if already_normalized:

        def f(x):
            return model(x)

    else:

        def f(x):
            return model(normalizer(x))

    return f


@torch.no_grad()
def predict_labels(
    forward_fn: Callable, x: torch.Tensor, batch_size: int = 256, device: str = "cuda"
):
    """Predict labels (argmax) and max softmax scores for a batch or dataset tensor x in [0,1]."""
    was_4d = x.dim() == 4
    if not was_4d:  # ensure NCHW
        x = x.unsqueeze(0)
    preds, confs = [], []
    for i in range(0, x.size(0), batch_size):
        xb = x[i : i + batch_size].to(device)
        logits = forward_fn(xb)
        pb = F.softmax(logits, dim=1)
        c, y = torch.max(pb, dim=1)
        preds.append(y.detach().cpu())
        confs.append(c.detach().cpu())
    preds = torch.cat(preds)
    confs = torch.cat(confs)
    if not was_4d:
        return preds[0].item(), confs[0].item()
    return preds, confs


def clamp01(t: torch.Tensor) -> torch.Tensor:
    return torch.clamp(t, 0.0, 1.0)


# -------------------------
# Property transforms (single-image helpers, CPU tensors expected; return CPU tensor in [0,1])
# -------------------------
def apply_brightness_delta(img: torch.Tensor, delta: float, direction: str) -> torch.Tensor:
    # img: 3x32x32 in [0,1]; direction in {"lighten","darken"}
    factor = 1.0 + float(delta) if direction == "lighten" else max(0.0, 1.0 - float(delta))
    out = TF.adjust_brightness(img, factor)
    return clamp01(out)


def apply_gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(img) * float(sigma)
    return clamp01(img + noise)


def apply_gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    # kernel size 5 keeps it sane for 32x32
    ks = 5
    return TF.gaussian_blur(img, kernel_size=[ks, ks], sigma=[float(sigma), float(sigma)])


_to_pil = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


def apply_jpeg(img: torch.Tensor, quality: int) -> torch.Tensor:
    # quality in [5..100]
    q = int(max(5, min(100, quality)))
    pil = _to_pil(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    pil2 = Image.open(buf).convert("RGB")
    out = _to_tensor(pil2)
    return clamp01(out)


# -------------------------
# Threshold search (binary search on a monotone parameter)
# -------------------------
def _predict_single(forward_fn: Callable, x_img_cpu: torch.Tensor, device: str) -> int:
    with torch.no_grad():
        logits = forward_fn(x_img_cpu.unsqueeze(0).to(device))
        return int(logits.argmax(1).item())


def find_break_threshold_binary(
    forward_fn: Callable,
    x0_cpu: torch.Tensor,
    y_ref: int,
    apply_fn: Callable[[torch.Tensor, float], torch.Tensor],
    lo: float,
    hi: float,
    steps: int,
    device: str,
) -> Tuple[float, bool]:
    """
    Returns (threshold, broke) where threshold is the minimal parameter (within [lo, hi])
    that flips prediction away from y_ref. If it never flips up to hi, returns (hi, False).
    """
    # Check at hi
    y_hi = _predict_single(forward_fn, apply_fn(x0_cpu, hi), device)
    if y_hi == y_ref:
        return hi, False
    left, right = lo, hi
    for _ in range(steps):
        mid = (left + right) / 2.0
        y_mid = _predict_single(forward_fn, apply_fn(x0_cpu, mid), device)
        if y_mid != y_ref:
            right = mid
        else:
            left = mid
    return right, True


def brightness_threshold(
    forward_fn, x_cpu, y_ref, delta_max: float, steps: int, device: str
) -> Tuple[float, bool]:
    t_li, broke_li = find_break_threshold_binary(
        forward_fn,
        x_cpu,
        y_ref,
        lambda im, d: apply_brightness_delta(im, d, "lighten"),
        0.0,
        delta_max,
        steps,
        device,
    )
    t_da, broke_da = find_break_threshold_binary(
        forward_fn,
        x_cpu,
        y_ref,
        lambda im, d: apply_brightness_delta(im, d, "darken"),
        0.0,
        delta_max,
        steps,
        device,
    )
    # pick smaller threshold among the directions that actually broke; if none broke, keep max
    candidates = []
    if broke_li:
        candidates.append(t_li)
    if broke_da:
        candidates.append(t_da)
    if candidates:
        return min(candidates), True
    # neither direction broke within bounds
    return min(t_li, t_da), False


# -------------------------
# Adversarial attack (PGD-Linf)
# -------------------------
def pgd_linf(
    model: nn.Module,
    normalizer: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    restarts: int = 1,
    device: str = "cuda",
    already_normalized: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (x_adv, success_mask). success_mask[i] True if attack found adversarial for sample i.
    """
    B = x.size(0)
    x = x.to(device)
    y = y.to(device)
    best_adv = x.clone()
    best_success = torch.zeros(B, dtype=torch.bool, device=device)

    def forward(inp):
        return model(inp) if already_normalized else model(normalizer(inp))

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for _ in range(restarts):
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        x_adv = clamp01(x + delta).detach().clone()
        x_adv.requires_grad_(True)

        for _ in range(steps):
            logits = forward(x_adv)
            loss = loss_fn(logits, y).sum()
            loss.backward()
            with torch.no_grad():
                grad = x_adv.grad
                x_adv = x_adv + alpha * torch.sign(grad)
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)  # project
                x_adv = clamp01(x_adv)
            x_adv.requires_grad_(True)

        with torch.no_grad():
            preds = forward(x_adv).argmax(1)
            success = preds != y
            # update best success (we only track success mask; best_adv is optional)
            improved = success & (~best_success)
            best_success = best_success | success
            best_adv[improved] = x_adv[improved]

    return best_adv.detach(), best_success.detach()


# -------------------------
# CIFAR-10-C helpers
# -------------------------
def load_cifar10c_corruption(root: str, name: str) -> np.ndarray:
    """
    Returns array of shape (50000, 32, 32, 3) for the given corruption.
    Blocks of 10000 correspond to severities 1..5.
    """
    path = os.path.join(root, f"{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CIFAR-10-C corruption not found: {path}")
    X = np.load(path)
    if X.shape[0] != 50000:
        raise ValueError(f"Unexpected shape for {name}: {X.shape}; expected (50000, 32, 32, 3).")
    return X


def load_cifar10c_labels(root: str) -> np.ndarray:
    path = os.path.join(root, "labels.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CIFAR-10-C labels not found: {path}")
    y = np.load(path)
    if y.shape[0] != 10000:
        raise ValueError(f"Unexpected labels shape: {y.shape}; expected (10000,).")
    return y


def np_to_tensor_images(arr: np.ndarray) -> torch.Tensor:
    """
    arr: (N, 32, 32, 3), uint8 or float
    returns: (N, 3, 32, 32) float in [0,1]
    """
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


# -------------------------
# Correlations (Spearman) without SciPy dependency
# -------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    # average ranks for ties
    temp = x.argsort(kind="mergesort")
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(1, len(x) + 1, dtype=float)
    # handle ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = sums / counts
    return avg[inv]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


# -------------------------
# Runner
# -------------------------
class Experiment2Runner:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        self.normalizer = get_normalize()
        self.model = get_model(self.device).to(self.device).eval()
        self.forward = make_forward(self.model, self.normalizer, args.dataset_returns_normalized)

    def compute_clean_preds(self, loader: DataLoader, n_eval: int):
        ys = []
        preds = []
        confs = []
        xs_buf = []
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                if total >= n_eval:
                    break
                take = min(xb.size(0), n_eval - total)
                xb = xb[:take]
                yb = yb[:take]
                logits = self.forward(xb.to(self.device))
                pb = F.softmax(logits, dim=1)
                c, yhat = torch.max(pb, dim=1)
                xs_buf.append(xb.cpu())
                ys.append(yb.cpu())
                preds.append(yhat.cpu())
                confs.append(c.cpu())
                total += take
        x_all = torch.cat(xs_buf, dim=0)  # [N,3,32,32] in [0,1] unless dataset already normalized
        y_all = torch.cat(ys, dim=0)
        pred_all = torch.cat(preds, dim=0)
        conf_all = torch.cat(confs, dim=0)
        return x_all, y_all, pred_all, conf_all

    def thresholds_for_batch(
        self, x_batch: torch.Tensor, y_ref_batch: torch.Tensor
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns dict mapping property -> (thresholds_raw, broke_flags)
        thresholds are 'raw' in property units, NOT normalized to [0,1].
        """
        args = self.args
        device = self.device
        B = x_batch.size(0)
        th_bright = np.zeros(B, dtype=np.float32)
        br_bright = np.zeros(B, dtype=bool)
        th_noise = np.zeros(B, dtype=np.float32)
        br_noise = np.zeros(B, dtype=bool)
        th_blur = np.zeros(B, dtype=np.float32)
        br_blur = np.zeros(B, dtype=bool)
        th_jpegS = np.zeros(B, dtype=np.float32)  # store JPEG "severity" = 100 - quality
        br_jpeg = np.zeros(B, dtype=bool)

        for i in range(B):
            x0 = x_batch[i].cpu()
            yref = int(y_ref_batch[i].item())
            # BRIGHTNESS (delta in [0, args.brightness_max])
            t_b, b_b = brightness_threshold(
                self.forward, x0, yref, args.brightness_max, args.binsearch_steps, device
            )
            th_bright[i] = t_b
            br_bright[i] = b_b

            # NOISE (sigma in [0, args.noise_max])
            t_n, b_n = find_break_threshold_binary(
                self.forward,
                x0,
                yref,
                apply_gaussian_noise,
                0.0,
                args.noise_max,
                args.binsearch_steps,
                device,
            )
            th_noise[i] = t_n
            br_noise[i] = b_n

            # BLUR (sigma in [0, args.blur_max])
            t_bl, b_bl = find_break_threshold_binary(
                self.forward,
                x0,
                yref,
                apply_gaussian_blur,
                0.0,
                args.blur_max,
                args.binsearch_steps,
                device,
            )
            th_blur[i] = t_bl
            br_blur[i] = b_bl

            # JPEG (quality from 100 -> 5). We search over a unit interval t∈[0,1], map to quality.
            def apply_jpeg_param(img, t):
                q = int(round(100 - t * (100 - 5)))  # t=0 => 100, t=1 => 5
                return apply_jpeg(img, q)

            t_jpeg_unit, b_j = find_break_threshold_binary(
                self.forward, x0, yref, apply_jpeg_param, 0.0, 1.0, args.binsearch_steps, device
            )
            # Convert to "severity" = 100 - quality_at_break
            q_break = int(round(100 - t_jpeg_unit * (100 - 5)))
            severity = 100 - q_break
            th_jpegS[i] = float(severity)
            br_jpeg[i] = b_j

        return {
            "brightness": (th_bright, br_bright),
            "noise": (th_noise, br_noise),
            "blur": (th_blur, br_blur),
            "jpeg": (th_jpegS, br_jpeg),
        }

    def run_pgd(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        _, success = pgd_linf(
            self.model,
            self.normalizer,
            x,
            y,
            eps=self.args.pgd_eps,
            alpha=self.args.pgd_alpha,
            steps=self.args.pgd_steps,
            restarts=self.args.pgd_restarts,
            device=self.device,
            already_normalized=self.args.dataset_returns_normalized,
        )
        # robust indicator: 1 if attack failed (i.e., model remained correct), else 0
        return (~success).cpu().numpy().astype(np.int32)

    def cifar10c_flip_severity(
        self, x_clean_pred: np.ndarray, idxs: np.ndarray, root: str, corruption: str
    ) -> np.ndarray:
        """
        For each index i (corresponding to CIFAR-10 test order), return minimal severity in 1..5
        where prediction differs from clean prediction; if never flips, return 6.
        """
        Xc = load_cifar10c_corruption(root, corruption)  # (50000,32,32,3)
        # precompute predictions for s=1..5
        preds_by_s = {}
        for s in range(1, 6):
            start = (s - 1) * 10000
            end = s * 10000
            arr = Xc[start:end]  # (10000,32,32,3)
            xt = np_to_tensor_images(arr)  # [10000,3,32,32] in [0,1]
            # only evaluate the subset of idxs to save time/memory
            xt = xt[idxs]
            with torch.no_grad():
                logits = self.forward(xt.to(self.device))
                preds = logits.argmax(1).detach().cpu().numpy()
            preds_by_s[s] = preds

        flip_sev = np.full(len(idxs), 6, dtype=np.int32)  # 6 means "no flip up to s=5"
        for j in range(len(idxs)):
            y0 = int(x_clean_pred[j])
            for s in range(1, 6):
                if int(preds_by_s[s][j]) != y0:
                    flip_sev[j] = s
                    break
        return flip_sev

    def run(self):
        args = self.args
        print(f"[Exp2] device={self.device}  n_eval={args.n_eval}")

        # 1) Load data & clean predictions
        test_loader = get_test_loader(
            batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
        )
        x_all, y_all, pred_all, conf_all = self.compute_clean_preds(test_loader, args.n_eval)

        # Optionally filter to images the model gets correct (common in robustness analysis)
        if args.only_correct:
            keep = pred_all == y_all
        else:
            keep = torch.ones_like(y_all, dtype=torch.bool)

        x = x_all[keep]
        y = y_all[keep]
        pred_clean = pred_all[keep]
        conf_clean = conf_all[keep]
        idxs = torch.arange(x_all.size(0))[keep].numpy()  # CIFAR-10 test indices

        print(
            f"[Exp2] Using {x.size(0)} images after filtering (only_correct={args.only_correct})."
        )

        # 2) Property thresholds
        thresholds = self.thresholds_for_batch(x, pred_clean)
        # Normalize thresholds to [0,1] for correlation comparability
        t_brightness_raw, _ = thresholds["brightness"]
        t_noise_raw, _ = thresholds["noise"]
        t_blur_raw, _ = thresholds["blur"]
        t_jpegS_raw, _ = thresholds["jpeg"]

        t_brightness = t_brightness_raw / max(1e-12, args.brightness_max)
        t_noise = t_noise_raw / max(1e-12, args.noise_max)
        t_blur = t_blur_raw / max(1e-12, args.blur_max)
        t_jpeg = t_jpegS_raw / 95.0  # severity range 0..95

        # 3) PGD robustness indicator (per image)
        robust_indicator = self.run_pgd(x, y)  # 1 robust, 0 broken

        # 4) CIFAR-10-C flip severities
        if args.cifar10c_root:
            sev_brightness = self.cifar10c_flip_severity(
                pred_clean.numpy(), idxs, args.cifar10c_root, "brightness"
            )
            sev_noise = self.cifar10c_flip_severity(
                pred_clean.numpy(), idxs, args.cifar10c_root, "gaussian_noise"
            )
            sev_blur = self.cifar10c_flip_severity(
                pred_clean.numpy(), idxs, args.cifar10c_root, "defocus_blur"
            )
            sev_jpeg = self.cifar10c_flip_severity(
                pred_clean.numpy(), idxs, args.cifar10c_root, "jpeg_compression"
            )
        else:
            sev_brightness = np.full(x.size(0), np.nan)
            sev_noise = np.full(x.size(0), np.nan)
            sev_blur = np.full(x.size(0), np.nan)
            sev_jpeg = np.full(x.size(0), np.nan)

        # 5) Correlations (Spearman)
        rho = {}
        rho["brightness_vs_pgd"] = spearman_corr(t_brightness, robust_indicator.astype(float))
        rho["noise_vs_pgd"] = spearman_corr(t_noise, robust_indicator.astype(float))
        rho["blur_vs_pgd"] = spearman_corr(t_blur, robust_indicator.astype(float))
        rho["jpeg_vs_pgd"] = spearman_corr(t_jpeg, robust_indicator.astype(float))

        rho["brightness_vs_c10c"] = spearman_corr(t_brightness, sev_brightness.astype(float))
        rho["noise_vs_c10c"] = spearman_corr(t_noise, sev_noise.astype(float))
        rho["blur_vs_c10c"] = spearman_corr(t_blur, sev_blur.astype(float))
        rho["jpeg_vs_c10c"] = spearman_corr(t_jpeg, sev_jpeg.astype(float))

        # 6) Save per-image table
        os.makedirs(args.out_dir, exist_ok=True)
        out_csv = os.path.join(args.out_dir, "exp2_results.csv")
        header = [
            "idx",
            "y_true",
            "y_pred_clean",
            "conf_clean",
            "thr_brightness_raw",
            "thr_noise_raw",
            "thr_blur_raw",
            "thr_jpeg_severity_raw",
            "thr_brightness_norm",
            "thr_noise_norm",
            "thr_blur_norm",
            "thr_jpeg_norm",
            "pgd_robust_indicator",
            "c10c_flip_brightness",
            "c10c_flip_gaussian_noise",
            "c10c_flip_defocus_blur",
            "c10c_flip_jpeg",
        ]
        with open(out_csv, "w") as f:
            f.write(",".join(header) + "\n")
            for i in range(x.size(0)):
                row = [
                    str(int(idxs[i])),
                    str(int(y[i].item())),
                    str(int(pred_clean[i].item())),
                    f"{float(conf_clean[i].item()):.6f}",
                    f"{float(t_brightness_raw[i]):.6f}",
                    f"{float(t_noise_raw[i]):.6f}",
                    f"{float(t_blur_raw[i]):.6f}",
                    f"{float(t_jpegS_raw[i]):.6f}",
                    f"{float(t_brightness[i]):.6f}",
                    f"{float(t_noise[i]):.6f}",
                    f"{float(t_blur[i]):.6f}",
                    f"{float(t_jpeg[i]):.6f}",
                    str(int(robust_indicator[i])),
                    str(int(sev_brightness[i])) if np.isfinite(sev_brightness[i]) else "",
                    str(int(sev_noise[i])) if np.isfinite(sev_noise[i]) else "",
                    str(int(sev_blur[i])) if np.isfinite(sev_blur[i]) else "",
                    str(int(sev_jpeg[i])) if np.isfinite(sev_jpeg[i]) else "",
                ]
                f.write(",".join(row) + "\n")
        print(f"[Exp2] Wrote per-image results to {out_csv}")

        # 7) Save summary
        summary = {
            "n_images": int(x.size(0)),
            "only_correct": bool(self.args.only_correct),
            "pgd": {
                "eps": self.args.pgd_eps,
                "alpha": self.args.pgd_alpha,
                "steps": self.args.pgd_steps,
                "restarts": self.args.pgd_restarts,
                "robust_rate": float(np.mean(robust_indicator)),
            },
            "correlations_spearman": rho,
            "property_maxima": {
                "brightness_max": self.args.brightness_max,
                "noise_max": self.args.noise_max,
                "blur_max": self.args.blur_max,
                "jpeg_severity_max": 95,
            },
        }
        out_json = os.path.join(args.out_dir, "exp2_summary.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Exp2] Wrote summary to {out_json}")
        print("[Exp2] Spearman correlations:")
        for k, v in rho.items():
            print(f"   {k:>24}: {v:+.4f}")

        return {"summary": summary, "csv": out_csv, "json": out_json}


# -------------------------
# CLI
# -------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Experiment 2: property thresholds vs robustness correlations"
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument(
        "--n_eval",
        type=int,
        default=512,
        help="Number of CIFAR-10 test images to evaluate (in order).",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--only-correct",
        dest="only_correct",
        action="store_true",
        help="Filter to clean-correct images.",
    )
    p.add_argument("--no-only-correct", dest="only_correct", action="store_false")
    p.set_defaults(only_correct=True)

    # Property search ranges + precision
    p.add_argument(
        "--binsearch-steps", type=int, default=7, help="Binary search iterations per property."
    )
    p.add_argument(
        "--brightness-max",
        type=float,
        default=0.8,
        help="Max delta for brightness (factor 1±delta).",
    )
    p.add_argument(
        "--noise-max", type=float, default=0.5, help="Max sigma for Gaussian noise in [0,1] scale."
    )
    p.add_argument("--blur-max", type=float, default=3.0, help="Max sigma for Gaussian blur.")

    # PGD attack
    p.add_argument("--pgd-eps", type=float, default=8 / 255)
    p.add_argument("--pgd-alpha", type=float, default=2 / 255)
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--pgd-restarts", type=int, default=1)

    # CIFAR-10-C root
    p.add_argument(
        "--cifar10c-root", type=str, default="", help="Path to CIFAR-10-C (with *.npy files)."
    )

    # Dataset normalization toggle
    p.add_argument(
        "--dataset-returns-normalized",
        action="store_true",
        help="Set if your Exp1 test loader already returns normalized tensors.",
    )

    p.add_argument("--out-dir", type=str, default="exp2_out")
    p.add_argument("--seed", type=int, default=123)
    return p


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    runner = Experiment2Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
