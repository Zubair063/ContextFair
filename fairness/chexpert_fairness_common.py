from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.chexpert import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    EVAL_CLASS_INDICES,
    EVAL_FINDING_COLS,
    NUM_CLASSES,
    build_frontal_dataloaders,
)

DEFAULT_BACKBONE = "resnet50"
FAIRREAD_BACKBONE = "resnet50"
SUPPORTED_BACKBONES = ("densenet121", "resnet50", "efficientnet_b0", "vgg16", "vit_b_16")

# Sensitive attributes: sex (female=0, male=1), age (<=60=0, >60=1).
ATTR_CARDINALITIES: dict[str, int] = {"sex": 2, "age": 2}
DEMO_ATTR_CHOICES: tuple[str, ...] = ("sex", "age")
DEMOGRAPHIC_SUBGROUP_CHOICES = ("sex", "age", "sex_age")

# Outer checkpoint / finetune selection (method training). Local ERM baseline
# Phase 0 is only required when metric is min_improvement (or --baseline-only).
SELECTION_METRIC_CHOICES = ("overall_macro_auc", "min_improvement", "worst_subgroup")
DEFAULT_SELECTION_METRIC = "overall_macro_auc"

DEFAULT_LR_GRID: tuple[float, ...] = (1e-4, 5e-5, 2e-5)
DEFAULT_WD_GRID: tuple[float, ...] = (1e-4, 1e-5)
DEFAULT_BATCH_SIZE_GRID: tuple[int, ...] = (32, 64, 128, 256)

# Keys accepted from ERM finetune_summary_*.json for the local baseline.
ERM_HPARAM_KEYS: tuple[str, ...] = (
    "lr",
    "weight_decay",
    "batch_size",
    "demo_embed_dim",
    "epochs",
    "lr_backbone",
    "lr_head",
)

# One row per (sample x eval pathology) — matches CheXpert Fusion/ERM/Fairness.
PREDICTION_CSV_FIELDS = [
    "sample_id",
    "rel_path",
    "pathology",
    "pathology_idx",
    "true_label",
    "logit",
    "prediction_probability",
    "pred_class",
    "sex",
    "age",
    "model_name",
    "seed",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def amp_active(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp and device.type == "cuda")


def setup_cudnn_benchmark(enabled: bool, device: torch.device) -> None:
    if device.type == "cuda" and enabled:
        torch.backends.cudnn.benchmark = True


def make_grad_scaler(use_amp: bool, device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=amp_active(device, use_amp))


def autocast_ctx(use_amp: bool, device: torch.device):
    return torch.amp.autocast("cuda", enabled=amp_active(device, use_amp))


def parse_seeds(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def generate_random_seeds(num_seeds: int) -> tuple[int, ...]:
    return tuple(random.randint(0, 2**31 - 1) for _ in range(num_seeds))


def resolve_seeds(
    seeds_arg: str,
    num_random_seeds: int,
    seeds_file: Path | None,
    save_seeds_file: Path | None,
) -> tuple[int, ...]:
    seeds_arg = str(seeds_arg).strip()
    if seeds_arg.lower() != "random":
        seeds = parse_seeds(seeds_arg)
    else:
        load_path = seeds_file
        if load_path is None and save_seeds_file is not None and save_seeds_file.is_file():
            load_path = save_seeds_file
        if load_path is not None and load_path.is_file():
            payload = json.loads(load_path.read_text(encoding="utf-8"))
            seeds = tuple(int(x) for x in (payload["seeds"] if isinstance(payload, dict) else payload))
            print(f"Loaded shared seeds from {load_path}: {seeds}")
        else:
            seeds = generate_random_seeds(num_random_seeds)
            print(f"Generated random seeds: {seeds}")

    out_path = save_seeds_file or seeds_file
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"seeds": list(seeds), "num_seeds": len(seeds)}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote shared seeds to {out_path}")
    return seeds


def canonicalize_attr(attr: str) -> str:
    a = str(attr).strip().lower()
    if a in ("gender", "sex"):
        return "sex"
    if a == "age":
        return "age"
    raise ValueError(f"Unknown demographic attr {attr!r}; use sex|gender|age")


def parse_demo_attrs(s: str) -> tuple[str, ...]:
    attrs = tuple(canonicalize_attr(x) for x in s.split(",") if x.strip())
    for attr in attrs:
        if attr not in ATTR_CARDINALITIES:
            raise ValueError("demographic attrs must be from: sex,age")
    if not attrs:
        raise ValueError("at least one demographic attribute is required")
    seen: set[str] = set()
    out: list[str] = []
    for a in attrs:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return tuple(out)


def parse_demo_subgroup(subgroup: str) -> tuple[str, ...]:
    key = subgroup.strip().lower().replace("-", "_")
    presets: dict[str, tuple[str, ...]] = {
        "sex": ("sex",),
        "gender": ("sex",),
        "age": ("age",),
        "sex_age": ("sex", "age"),
        "age_sex": ("sex", "age"),
        "gender_age": ("sex", "age"),
        "age_gender": ("sex", "age"),
    }
    if key not in presets:
        raise ValueError(
            f"Unknown demographic subgroup '{subgroup}'. "
            f"Choose from: {', '.join(DEMOGRAPHIC_SUBGROUP_CHOICES)}"
        )
    return presets[key]


def resolve_demo_attrs(subgroup: str | None, attrs: str) -> tuple[str, ...]:
    if subgroup:
        return parse_demo_subgroup(subgroup)
    if attrs.strip():
        return parse_demo_attrs(attrs)
    return ("sex",)


def demographic_run_suffix(subgroup: str | None, attrs: tuple[str, ...]) -> str:
    if subgroup:
        return subgroup.strip().lower().replace("-", "_")
    return "_".join(attrs)


def normalize_demographic_subgroup(subgroup: str | None) -> str | None:
    if not subgroup:
        return None
    return subgroup.strip().lower().replace("-", "_")


def run_group_name(method: str, demo_attrs: tuple[str, ...], subgroup: str | None = None) -> str:
    """e.g. GroupDRO_sex, FairDRO_age, FairREAD_sex_age."""
    return f"{method}_{demographic_run_suffix(subgroup, demo_attrs)}"


def get_checkpoint_root(
    output_dir: Path | str,
    method: str,
    demo_attrs: tuple[str, ...],
    subgroup: str | None = None,
) -> Path:
    return Path(output_dir) / "checkpoints" / run_group_name(method, demo_attrs, subgroup)


def get_results_root(
    output_dir: Path | str,
    method: str,
    demo_attrs: tuple[str, ...],
    subgroup: str | None = None,
) -> Path:
    return Path(output_dir) / "results" / run_group_name(method, demo_attrs, subgroup)


def get_output_roots(
    output_dir: Path | str,
    method: str,
    demo_attrs: tuple[str, ...],
    subgroup: str | None = None,
) -> tuple[Path, Path]:
    return (
        get_checkpoint_root(output_dir, method, demo_attrs, subgroup),
        get_results_root(output_dir, method, demo_attrs, subgroup),
    )


def apply_demographic_config(cfg: Any, subgroup: str | None, attrs: str) -> str:
    cfg.demographic_subgroup = normalize_demographic_subgroup(subgroup)
    cfg.demographic_attrs = resolve_demo_attrs(subgroup, attrs)
    return demographic_run_suffix(cfg.demographic_subgroup, cfg.demographic_attrs)


def add_demographic_arguments(parser: argparse.ArgumentParser, default_attrs: str = "sex") -> None:
    parser.add_argument(
        "--demographic-subgroup",
        type=str,
        default=None,
        choices=list(DEMOGRAPHIC_SUBGROUP_CHOICES),
        help="Preset: sex, age, sex_age",
    )
    parser.add_argument(
        "--demographic-attrs",
        type=str,
        default=default_attrs,
        help="Comma-separated: sex, age, or sex,age (gender is an alias for sex)",
    )


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: str,
    default_backbone: str = DEFAULT_BACKBONE,
    default_epochs: int = 20,
    default_batch_size: int = 128,
) -> None:
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--backbone", type=str, default=default_backbone)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seeds", type=str, default="random")
    parser.add_argument("--num-random-seeds", type=int, default=5)
    parser.add_argument("--seeds-file", type=str, default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-use-amp", action="store_true")
    add_demographic_arguments(parser)


def build_dataloaders(cfg: Any) -> tuple[DataLoader, DataLoader, DataLoader]:
    return build_frontal_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        pin_memory=True,
        aug_style="erm",
    )


def eval_train_loader(train_loader: DataLoader, cfg: Any) -> DataLoader:
    """Non-shuffled loader over the training set (for per-epoch subgroup stats)."""
    kw: dict[str, Any] = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    if cfg.num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(train_loader.dataset, **kw)


def compute_n_groups(demo_attrs: tuple[str, ...]) -> int:
    n = 1
    for attr in demo_attrs:
        n *= ATTR_CARDINALITIES[attr]
    return n


def compute_sa_dim(demo_attrs: tuple[str, ...]) -> int:
    """Total input dimension for a sensitive-attribute encoder (FairREAD)."""
    dim = 0
    for attr in demo_attrs:
        card = ATTR_CARDINALITIES[attr]
        dim += 1 if card == 2 else card
    return dim


def build_group_ids(batch: dict, demo_attrs: tuple[str, ...], device: torch.device) -> torch.Tensor:
    vals = []
    cards = []
    for attr in demo_attrs:
        v = batch[attr].to(device).long().clamp(min=0)
        vals.append(v)
        cards.append(ATTR_CARDINALITIES[attr])

    group_ids = torch.zeros_like(vals[0])
    for i, v in enumerate(vals):
        stride = 1
        for c in cards[i + 1 :]:
            stride *= c
        group_ids = group_ids + v * stride
    return group_ids


def build_joint_group_ids(
    labels: torch.Tensor,
    sensitive_ids: torch.Tensor,
    n_classes: int = 2,
) -> torch.Tensor:
    return sensitive_ids * n_classes + labels.long()


def flatten_multilabel(
    tensor: torch.Tensor,
    sensitive_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, L) -> (B*L,) with sensitive_ids repeated per label column."""
    b, length = tensor.shape
    flat = tensor.reshape(-1)
    groups = sensitive_ids.unsqueeze(1).expand(b, length).reshape(-1)
    return flat, groups


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    from torchvision.models import (
        DenseNet121_Weights,
        EfficientNet_B0_Weights,
        ResNet50_Weights,
        VGG16_Weights,
        ViT_B_16_Weights,
        densenet121,
        efficientnet_b0,
        resnet50,
        vgg16,
        vit_b_16,
    )

    name = name.lower()
    if name == "resnet50":
        w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = resnet50(weights=w)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim
    if name == "efficientnet_b0":
        w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b0(weights=w)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    if name == "densenet121":
        w = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = densenet121(weights=w)
        feat_dim = m.classifier.in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    if name == "vit_b_16":
        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = vit_b_16(weights=w)
        feat_dim = m.heads.head.in_features
        m.heads = nn.Identity()
        return m, feat_dim
    if name == "vgg16":
        w = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        m = vgg16(weights=w)
        feat_dim = m.classifier[6].in_features
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
        return m, feat_dim
    raise ValueError(f"Unsupported backbone '{name}'. Choose from: {SUPPORTED_BACKBONES}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _commit_pending_dir(pending: Path, final: Path) -> None:
    if final.exists():
        shutil.rmtree(final)
    pending.rename(final)


def save_seed_checkpoint(
    ckpt_root: Path,
    seed: int,
    state: Any,
    ckpt_filename: str,
) -> Path:
    ckpt_root.mkdir(parents=True, exist_ok=True)
    pending_ckpt = ckpt_root / f"_pending_seed_{seed}"
    final_ckpt = ckpt_root / f"seed_{seed}"
    if pending_ckpt.exists():
        shutil.rmtree(pending_ckpt)
    pending_ckpt.mkdir(parents=True)
    try:
        torch.save(state, pending_ckpt / ckpt_filename)
        _commit_pending_dir(pending_ckpt, final_ckpt)
        return final_ckpt / ckpt_filename
    except Exception:
        shutil.rmtree(pending_ckpt, ignore_errors=True)
        raise


def save_seed_results(results_root: Path, seed: int, rows: list[dict[str, Any]]) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    pending_results = results_root / f"_pending_seed_{seed}"
    final_results = results_root / f"seed_{seed}"
    if pending_results.exists():
        shutil.rmtree(pending_results)
    pending_results.mkdir(parents=True)
    try:
        csv_path = pending_results / "test_predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PREDICTION_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        _commit_pending_dir(pending_results, final_results)
        return final_results / "test_predictions.csv"
    except Exception:
        shutil.rmtree(pending_results, ignore_errors=True)
        raise


def persist_seed_run(
    results_root: Path,
    ckpt_root: Path,
    seed: int,
    state: Any,
    ckpt_filename: str,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    ckpt_path = save_seed_checkpoint(ckpt_root, seed, state, ckpt_filename)
    csv_path = save_seed_results(results_root, seed, rows)
    return csv_path, ckpt_path


@torch.no_grad()
def collect_test_rows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    seed: int,
    use_amp: bool = False,
    *,
    forward_fn: Callable[[nn.Module, dict, torch.device], torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """One row per (sample x eval pathology), matching CheXpert ERM/Fusion CSV schema."""
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with autocast_ctx(use_amp, device):
            logits = forward_fn(model, batch, device) if forward_fn is not None else model(x)
        logits = logits.float()
        probs = torch.sigmoid(logits)
        pred_cls = (probs >= 0.5).long()

        sample_ids = batch["sample_id"]
        rel_paths = batch["rel_path"]
        sexes = batch["sex"].tolist()
        ages = batch["age"].tolist()
        labels_cpu = y.cpu()
        logits_cpu = logits.cpu()
        probs_cpu = probs.cpu()
        pred_cpu = pred_cls.cpu()

        for i in range(len(sample_ids)):
            for pathology, j in zip(EVAL_FINDING_COLS, EVAL_CLASS_INDICES):
                rows.append(
                    {
                        "sample_id": sample_ids[i],
                        "rel_path": rel_paths[i],
                        "pathology": pathology,
                        "pathology_idx": int(j),
                        "true_label": int(labels_cpu[i, j].item()),
                        "logit": float(logits_cpu[i, j].item()),
                        "prediction_probability": float(probs_cpu[i, j].item()),
                        "pred_class": int(pred_cpu[i, j].item()),
                        "sex": int(sexes[i]),
                        "age": int(ages[i]),
                        "model_name": model_name,
                        "seed": int(seed),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Macro-AUC metrics (5 competition pathologies) — matches Fusion/ERM.
# ---------------------------------------------------------------------------

def eval_subset(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.tensor(EVAL_CLASS_INDICES, device=logits.device, dtype=torch.long)
    return logits.index_select(1, idx), labels.index_select(1, idx)


def macro_auc_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_indices: tuple[int, ...] = EVAL_CLASS_INDICES,
) -> float:
    """Mean per-pathology AUC over the given label columns (default: 5-pathology eval set)."""
    probs = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy()
    aucs: list[float] = []
    for j in class_indices:
        if len(np.unique(y[:, j])) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y[:, j], probs[:, j])))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


def subgroup_macro_auc_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    subgroups: torch.Tensor,
    num_subgroups: int,
) -> dict[str, float]:
    """Overall + per-subgroup macro AUC on the 5 eval pathologies."""
    report: dict[str, float] = {
        "overall_macro_auc": macro_auc_from_logits(logits, labels),
        "overall_macro_auc_13": macro_auc_from_logits(
            logits, labels, class_indices=tuple(range(NUM_CLASSES))
        ),
    }
    subgroups_np = subgroups.cpu().numpy()
    for g in range(num_subgroups):
        mask = subgroups_np == g
        if int(mask.sum()) < 2:
            report[f"subgroup_{g}_macro_auc"] = float("nan")
            continue
        report[f"subgroup_{g}_macro_auc"] = macro_auc_from_logits(logits[mask], labels[mask])
    return report


def worst_subgroup_macro_auc(report: dict[str, float], num_subgroups: int) -> float:
    vals = [float(report.get(f"subgroup_{g}_macro_auc", float("nan"))) for g in range(num_subgroups)]
    vals = [v for v in vals if not math.isnan(v)]
    return min(vals) if vals else float("nan")


def min_subgroup_macro_auc_improvement(
    base_report: dict[str, float], compare_report: dict[str, float], num_subgroups: int
) -> float:
    deltas = []
    for g in range(num_subgroups):
        b = float(base_report.get(f"subgroup_{g}_macro_auc", float("nan")))
        c = float(compare_report.get(f"subgroup_{g}_macro_auc", float("nan")))
        if not math.isnan(b) and not math.isnan(c):
            deltas.append(c - b)
    return float(min(deltas)) if deltas else float("nan")


@torch.no_grad()
def evaluate_subgroup_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    selection_attr: str,
    use_amp: bool = False,
    *,
    forward_fn: Callable[[nn.Module, dict, torch.device], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Evaluate 5-path val loss + overall/subgroup macro AUC on ``selection_attr``."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    logits_all, labels_all, groups_all = [], [], []
    loss_sum, n = 0.0, 0
    num_subgroups = ATTR_CARDINALITIES[selection_attr]

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        g = batch[selection_attr].to(device, non_blocking=True).clamp(min=0)
        with autocast_ctx(use_amp, device):
            logits = forward_fn(model, batch, device) if forward_fn is not None else model(x)
            eval_logits, eval_labels = eval_subset(logits, y)
            loss = criterion(eval_logits.float(), eval_labels)
        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs
        logits_all.append(logits.float().cpu())
        labels_all.append(y.cpu())
        groups_all.append(g.cpu())

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    groups = torch.cat(groups_all)
    report = subgroup_macro_auc_report(logits, labels, groups, num_subgroups)
    report["val_loss"] = loss_sum / max(n, 1)
    report["worst_subgroup_macro_auc"] = worst_subgroup_macro_auc(report, num_subgroups)
    return report


# ---------------------------------------------------------------------------
# Fusion-style grids / selection / ERM baselines / sequential finetune
# ---------------------------------------------------------------------------

def parse_float_grid(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def parse_int_grid(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def mid_of(grid: Sequence[Any], fallback: Any) -> Any:
    if not grid:
        return fallback
    return grid[len(grid) // 2]


def add_finetune_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--erm-hparams-dir",
        type=str,
        default="",
        help="Dir with finetune_summary_ERM__{model}.json (ERM finetune outputs)",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default=DEFAULT_SELECTION_METRIC,
        choices=list(SELECTION_METRIC_CHOICES),
        help=(
            "Checkpoint / finetune selection: overall_macro_auc (default), "
            "min_improvement (vs local ERM), or worst_subgroup"
        ),
    )
    parser.add_argument(
        "--selection-attr",
        type=str,
        default="",
        help="sex|age for worst_subgroup / min_improvement subgroup splits",
    )
    parser.add_argument(
        "--baseline-selection-attr",
        type=str,
        default="sex",
        help="sex|age for baseline worst-subgroup",
    )
    parser.add_argument("--lr-grid", type=str, default=",".join(str(x) for x in DEFAULT_LR_GRID))
    parser.add_argument(
        "--weight-decay-grid", type=str, default=",".join(str(x) for x in DEFAULT_WD_GRID)
    )
    parser.add_argument(
        "--batch-size-grid", type=str, default=",".join(str(x) for x in DEFAULT_BATCH_SIZE_GRID)
    )
    parser.add_argument("--finetune-summary", type=str, default="")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--finetune-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")


def resolve_selection_attr(explicit: str, demo_attrs: tuple[str, ...]) -> str:
    if explicit.strip():
        attr = canonicalize_attr(explicit.strip())
    else:
        attr = demo_attrs[0] if demo_attrs else "sex"
    if attr not in DEMO_ATTR_CHOICES:
        raise ValueError(f"selection_attr must be sex|age; got {attr!r}")
    return attr


def normalize_selection_metric(metric: str) -> str:
    key = str(metric).strip().lower().replace("-", "_")
    aliases = {
        "min_improvement_gap": "min_improvement",
        "improvement_gap": "min_improvement",
        "min_gap": "min_improvement",
        "overall": "overall_macro_auc",
        "overall_auc": "overall_macro_auc",
        "macro_auc": "overall_macro_auc",
        "auc": "overall_macro_auc",
        "worst": "worst_subgroup",
        "worst_auc": "worst_subgroup",
        "worst_macro_auc": "worst_subgroup",
    }
    key = aliases.get(key, key)
    if key not in SELECTION_METRIC_CHOICES:
        raise ValueError(
            f"selection_metric must be one of {SELECTION_METRIC_CHOICES}; got {metric!r}"
        )
    return key


def needs_local_baseline(selection_metric: str) -> bool:
    return normalize_selection_metric(selection_metric) == "min_improvement"


def compute_selection_score(
    report: dict[str, float],
    *,
    selection_metric: str,
    selection_attr: str,
    baseline_report: dict[str, float] | None = None,
) -> float:
    """Score used for finetune trials and epoch checkpointing."""
    metric = normalize_selection_metric(selection_metric)
    if metric == "overall_macro_auc":
        return float(report.get("overall_macro_auc", float("nan")))
    if metric == "worst_subgroup":
        if "worst_subgroup_macro_auc" in report:
            return float(report["worst_subgroup_macro_auc"])
        return worst_subgroup_macro_auc(report, ATTR_CARDINALITIES[selection_attr])
    if metric == "min_improvement":
        if baseline_report is None:
            raise ValueError(
                "selection_metric=min_improvement requires a local ERM baseline_report"
            )
        return min_subgroup_macro_auc_improvement(
            baseline_report, report, ATTR_CARDINALITIES[selection_attr]
        )
    raise ValueError(f"Unhandled selection_metric={metric!r}")


def maybe_load_baseline_report(
    cfg: Any,
    model_name: str,
    seed: int,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float] | None:
    if not needs_local_baseline(getattr(cfg, "selection_metric", DEFAULT_SELECTION_METRIC)):
        return None
    return load_baseline_report(
        cfg.output_dir,
        model_name,
        int(seed),
        val_loader,
        device,
        cfg.selection_attr,
        use_amp=bool(getattr(cfg, "use_amp", False)),
        pretrained=bool(getattr(cfg, "pretrained", True)),
    )


class ImageOnlyERM(nn.Module):
    """Image-only multilabel ERM used as local baseline for min-improvement selection."""

    def __init__(self, backbone_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def train_erm_baseline_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    selection_attr: str,
    use_amp: bool = False,
) -> tuple[nn.Module, dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = make_grad_scaler(use_amp, device)
    criterion = nn.BCEWithLogitsLoss()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_report: dict[str, float] = {}

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Baseline {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp, device):
                logits = model(x)
                loss = criterion(logits, y)  # full 13-pathology BCE
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_report = evaluate_subgroup_report(
            model, val_loader, device, selection_attr, use_amp=use_amp
        )
        score = float(val_report["worst_subgroup_macro_auc"])
        val_report["selection_score"] = score if not math.isnan(score) else float("nan")
        print(
            f"Baseline epoch {epoch + 1:03d}: val_loss={val_report['val_loss']:.4f} "
            f"macro_auc={val_report['overall_macro_auc']:.4f} "
            f"worst_sub={val_report['worst_subgroup_macro_auc']:.4f}"
        )
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_report = dict(val_report)

    model.load_state_dict(best_state)
    if not best_report:
        best_report = evaluate_subgroup_report(
            model, val_loader, device, selection_attr, use_amp=use_amp
        )
        best_report["selection_score"] = float(best_report["worst_subgroup_macro_auc"])
    return model, best_report


def image_only_ckpt_path(output_dir: Path | str, model_name: str, seed: int) -> Path:
    return Path(output_dir) / "checkpoints" / "ERM" / f"seed_{seed}" / f"{model_name}_image_only.pt"


def erm_hparams_summary_path(erm_hparams_dir: str, model_name: str) -> Path:
    root = Path(erm_hparams_dir) if str(erm_hparams_dir).strip() else Path()
    return root / f"finetune_summary_ERM__{model_name}.json"


def filter_erm_hparams(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ERM_HPARAM_KEYS:
        if key in raw:
            out[key] = raw[key]
    ignored = sorted(set(raw) - set(out))
    if ignored:
        print(f"Ignoring non-hparam keys from ERM summary: {ignored}")
    return out


def load_erm_baseline_hparams(erm_hparams_dir: str, model_name: str) -> dict[str, Any] | None:
    path = erm_hparams_summary_path(erm_hparams_dir, model_name)
    if not str(erm_hparams_dir).strip() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    hparams = payload.get("hparams")
    if not isinstance(hparams, dict):
        raise ValueError(f"No 'hparams' dict in ERM finetune summary: {path}")
    filtered = filter_erm_hparams(hparams)
    if not filtered:
        raise ValueError(f"ERM finetune summary has no usable hparams: {path}")
    print(
        f"Loaded ERM baseline hparams for {model_name} from {path}: {filtered} "
        f"(seeds come from method experiment_seeds, not ERM)"
    )
    return filtered


def apply_hparams_to_cfg(cfg: Any, hparams: dict[str, Any]) -> Any:
    cfg = copy.deepcopy(cfg)
    for key, value in hparams.items():
        if hasattr(cfg, key):
            cur = getattr(cfg, key)
            if isinstance(cur, int) and not isinstance(cur, bool):
                setattr(cfg, key, int(value))
            elif isinstance(cur, float):
                setattr(cfg, key, float(value))
            else:
                setattr(cfg, key, value)
    return cfg


def resolve_baseline_hparams(
    *,
    output_dir: Path | str,
    erm_hparams_dir: str,
    model_name: str,
    seeds: tuple[int, ...],
    baseline_selection_attr: str,
) -> dict[str, Any]:
    local_summary = Path(output_dir) / "finetune" / f"finetune_summary_ERM__{model_name}.json"
    erm_hparams = load_erm_baseline_hparams(erm_hparams_dir, model_name)
    if erm_hparams is not None:
        summary = {
            "run_group": "ERM",
            "model": model_name,
            "selection_metric": "worst_subgroup_macro_auc",
            "selection_attr": baseline_selection_attr,
            "hparams": erm_hparams,
            "source": str(erm_hparams_summary_path(erm_hparams_dir, model_name)),
            "note": (
                "Hyperparameters only from ERM finetune summary; "
                "baseline training uses method shared seeds."
            ),
            "method_seeds": list(seeds),
        }
        save_json(local_summary, summary)
        return erm_hparams

    if local_summary.is_file():
        payload = json.loads(local_summary.read_text(encoding="utf-8"))
        print(
            f"Using local baseline finetune summary (hparams only): {local_summary} "
            f"| method seeds={list(seeds)}"
        )
        return filter_erm_hparams(dict(payload["hparams"]))

    raise FileNotFoundError(
        f"ERM baseline hparams required for {model_name}.\n"
        f"Expected: {erm_hparams_summary_path(erm_hparams_dir, model_name)}\n"
        "Run CheXpert ERM image-only finetune first, then point "
        "--erm-hparams-dir at that finetune/ folder."
    )


def load_baseline_report(
    output_dir: Path | str,
    model_name: str,
    seed: int,
    val_loader: DataLoader,
    device: torch.device,
    selection_attr: str,
    *,
    use_amp: bool = False,
    pretrained: bool = True,
) -> dict[str, float]:
    ckpt = image_only_ckpt_path(output_dir, model_name, seed)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Image-only ERM baseline checkpoint missing: {ckpt}\n"
            "Run baseline training first or unset --skip-baseline."
        )
    baseline = ImageOnlyERM(model_name, pretrained=pretrained).to(device)
    baseline.load_state_dict(torch.load(ckpt, map_location=device))
    report = evaluate_subgroup_report(
        baseline, val_loader, device, selection_attr, use_amp=use_amp
    )
    print(
        f"Baseline ERM val: overall={report['overall_macro_auc']:.4f} "
        f"worst_sub={report['worst_subgroup_macro_auc']:.4f} ({ckpt.name})"
    )
    return report


def ensure_baseline_erm(cfg: Any, model_name: str, device: torch.device) -> None:
    """Train image-only multilabel ERM for all shared seeds using ERM hparams only."""
    missing = [
        s for s in cfg.seeds if not image_only_ckpt_path(cfg.output_dir, model_name, int(s)).is_file()
    ]
    if not missing:
        print(f"Baseline ERM for {model_name}: checkpoints exist for all seeds; skipping train.")
        return
    if getattr(cfg, "skip_baseline", False):
        raise FileNotFoundError(
            f"skip_baseline=1 but missing image-only ckpts for {model_name}, seeds={missing}"
        )

    print(f"\n########## Phase 0: image-only ERM baseline ({model_name}) ##########")
    print(f"Shared seeds: {cfg.seeds}")
    print(f"Missing seeds to train: {missing}")
    print(f"Baseline selection_attr={cfg.baseline_selection_attr} (worst_subgroup_macro_auc)")

    use_amp = bool(getattr(cfg, "use_amp", False))
    pretrained = bool(getattr(cfg, "pretrained", True))
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    hparams = resolve_baseline_hparams(
        output_dir=cfg.output_dir,
        erm_hparams_dir=cfg.erm_hparams_dir,
        model_name=model_name,
        seeds=cfg.seeds,
        baseline_selection_attr=cfg.baseline_selection_attr,
    )
    base_cfg = apply_hparams_to_cfg(cfg, hparams)
    print(f"Baseline hparams: {hparams}")
    if int(base_cfg.batch_size) != int(cfg.batch_size):
        train_loader, val_loader, test_loader = build_dataloaders(base_cfg)
        print(f"Using baseline batch_size={base_cfg.batch_size}")

    ckpt_root = Path(cfg.output_dir) / "checkpoints" / "ERM"
    results_root = Path(cfg.output_dir) / "results" / "ERM"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    for seed in cfg.seeds:
        seed_results_dir = results_root / f"seed_{seed}"
        seed_results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = seed_results_dir / "test_predictions.csv"
        if not csv_path.is_file():
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=PREDICTION_CSV_FIELDS).writeheader()

    for seed in cfg.seeds:
        ckpt_path = image_only_ckpt_path(base_cfg.output_dir, model_name, int(seed))
        if ckpt_path.is_file():
            print(f"Baseline exists, skip seed={seed}: {ckpt_path}")
            continue
        print("=" * 80)
        print(
            f"Train baseline ERM model={model_name} seed={seed} "
            f"lr={base_cfg.lr} wd={base_cfg.weight_decay} bs={base_cfg.batch_size}"
        )
        train_loader, val_loader, test_loader = build_dataloaders(base_cfg)
        set_seed(int(seed))
        model = ImageOnlyERM(model_name, pretrained=pretrained).to(device)
        model, best_report = train_erm_baseline_one_run(
            model,
            train_loader,
            val_loader,
            device,
            epochs=int(base_cfg.epochs),
            lr=float(base_cfg.lr),
            weight_decay=float(base_cfg.weight_decay),
            selection_attr=cfg.baseline_selection_attr,
            use_amp=use_amp,
        )
        seed_ckpt_dir = ckpt_path.parent
        seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        save_json(
            seed_ckpt_dir / f"{model_name}_image_only_val_metrics.json",
            {
                "seed": int(seed),
                "selection_metric": "worst_subgroup_macro_auc",
                "selection_attr": cfg.baseline_selection_attr,
                "hparams_source": "ERM_finetune_summary",
                "hparams": {
                    "lr": base_cfg.lr,
                    "weight_decay": base_cfg.weight_decay,
                    "batch_size": base_cfg.batch_size,
                    "epochs": base_cfg.epochs,
                },
                "val_metrics": best_report,
            },
        )
        print(f"Saved baseline checkpoint: {ckpt_path}")

        run_name = f"{model_name}_image_only"
        rows = collect_test_rows(model, test_loader, device, run_name, int(seed), use_amp=use_amp)
        csv_path = results_root / f"seed_{seed}" / "test_predictions.csv"
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=PREDICTION_CSV_FIELDS).writerows(rows)
        print(f"Updated baseline test CSV: {csv_path.resolve()}")


def sequential_finetune(
    *,
    stages: list[tuple[str, Sequence[Any]]],
    initial_hparams: dict[str, Any],
    trial_fn: Callable[[dict[str, Any], str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Coordinate-ascent hparam search. ``trial_fn(hparams, stage)`` must return
    a dict with ``selection_score`` and preferably ``hparams`` / ``metrics``.
    """
    best_hparams = dict(initial_hparams)
    trials: list[dict[str, Any]] = []

    for stage_name, grid in stages:
        if not grid:
            continue
        print(f"\n========== Finetune {stage_name} grid ==========")
        best_score = float("-inf")
        for value in grid:
            hp = {**best_hparams, stage_name: value}
            if stage_name in ("lr", "weight_decay") or "coeff" in stage_name or stage_name.startswith("lambda") or stage_name in (
                "rho",
                "gamma",
                "robust_step_size",
                "temperature",
                "adv_alpha",
                "fair_coeff",
            ):
                hp[stage_name] = float(value)
            elif stage_name in ("batch_size", "zdim", "mean_std_dim"):
                hp[stage_name] = int(value)
            entry = trial_fn(hp, stage_name)
            trials.append(entry)
            score = float(entry.get("selection_score", float("nan")))
            if not math.isnan(score) and score > best_score:
                best_score = score
                best_hparams = {**best_hparams, **entry.get("hparams", hp)}
        print(f"Best after {stage_name}: {best_hparams} (score={best_score:.4f})")

    return best_hparams, trials


def finetune_summary_path(
    output_dir: Path | str,
    run_group: str,
    model_name: str,
    explicit: str = "",
) -> Path:
    if str(explicit).strip():
        return Path(explicit)
    return Path(output_dir) / "finetune" / f"finetune_summary_{run_group}__{model_name}.json"


def load_or_run_finetune(
    *,
    cfg: Any,
    run_group: str,
    model_name: str,
    stages: list[tuple[str, Sequence[Any]]],
    initial_hparams: dict[str, Any],
    trial_fn: Callable[[dict[str, Any], str], dict[str, Any]],
    grids_meta: dict[str, list[Any]],
) -> dict[str, Any]:
    summary_path = finetune_summary_path(
        cfg.output_dir, run_group, model_name, getattr(cfg, "finetune_summary", "")
    )
    if getattr(cfg, "skip_finetune", False):
        if summary_path.is_file():
            print(f"skip_finetune=1; loading summary: {summary_path}")
            return json.loads(summary_path.read_text(encoding="utf-8"))
        print("skip_finetune=1 and no summary; using CLI/default hparams")
        return {
            "run_group": run_group,
            "model": model_name,
            "hparams": initial_hparams,
            "selection_metric": normalize_selection_metric(
                getattr(cfg, "selection_metric", DEFAULT_SELECTION_METRIC)
            ),
            "selection_attr": cfg.selection_attr,
            "skipped": True,
        }

    if summary_path.is_file() and not getattr(cfg, "finetune_only", False):
        print(f"Finetune summary exists, loading: {summary_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"\n########## Finetune {run_group}/{model_name} on seed {cfg.seeds[0]} ##########")
    best_hparams, trials = sequential_finetune(
        stages=stages,
        initial_hparams=initial_hparams,
        trial_fn=trial_fn,
    )
    summary = {
        "run_group": run_group,
        "model": model_name,
        "finetune_seed": int(cfg.seeds[0]),
        "selection_metric": normalize_selection_metric(
            getattr(cfg, "selection_metric", DEFAULT_SELECTION_METRIC)
        ),
        "selection_attr": cfg.selection_attr,
        "demographic_attrs": list(cfg.demographic_attrs),
        "epochs": int(getattr(cfg, "epochs", 0) or 0),
        "hparams": best_hparams,
        "grids": grids_meta,
        "trials": [
            {
                "stage": t.get("stage"),
                "hparams": t.get("hparams"),
                "selection_score": t.get("selection_score"),
                "overall_macro_auc": (t.get("metrics") or {}).get("overall_macro_auc"),
                "worst_subgroup_macro_auc": (t.get("metrics") or {}).get("worst_subgroup_macro_auc"),
            }
            for t in trials
        ],
    }
    save_json(summary_path, summary)
    print(f"Wrote finetune summary: {summary_path}")
    return summary
