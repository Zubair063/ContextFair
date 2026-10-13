from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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
    FINDING_COLS,
    NUM_CLASSES,
    build_frontal_dataloaders,
)

SUPPORTED_MODELS: tuple[str, ...] = (
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "vit_b_16",
    "vgg16",
)

# Canonical demographic attrs (gender is accepted as an alias for sex).
ATTR_CARDINALITIES: dict[str, int] = {
    "sex": 2,  # female=0, male=1
    "age": 2,  # <=60=0, >60=1
}
DEMO_ATTR_CHOICES: tuple[str, ...] = ("sex", "age")

DEFAULT_LR_GRID: tuple[float, ...] = (1e-4, 5e-5, 2e-5)
DEFAULT_WD_GRID: tuple[float, ...] = (1e-4, 1e-5, 2e-5)
DEFAULT_BATCH_SIZE_GRID: tuple[int, ...] = (32, 64, 128, 256)
DEFAULT_DEMO_DIM_GRID: tuple[int, ...] = (8, 16, 32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonicalize_attr(attr: str) -> str:
    a = str(attr).strip().lower()
    if a in ("gender", "sex"):
        return "sex"
    if a == "age":
        return "age"
    raise ValueError(f"Unknown demographic attr {attr!r}; use sex|gender|age")


@dataclass
class Config:
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = "chexpert_erm_outputs"
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-4
    lr_backbone: float = 0.0
    lr_head: float = 0.0
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224
    models: tuple[str, ...] = ("resnet50",)
    seeds: tuple[int, ...] = ()
    use_demographics: bool = False
    demographic_attrs: tuple[str, ...] = ()
    # Attr for worst-subgroup / min-improvement (defaults to first demo attr or sex).
    selection_attr: str = "sex"
    use_amp: bool = True
    demo_embed_dim: int = 16
    # Finetune grids (sequential: lr → weight_decay → batch_size → [demo_embed_dim])
    lr_grid: tuple[float, ...] = DEFAULT_LR_GRID
    weight_decay_grid: tuple[float, ...] = DEFAULT_WD_GRID
    batch_size_grid: tuple[int, ...] = DEFAULT_BATCH_SIZE_GRID
    demo_embed_dim_grid: tuple[int, ...] = DEFAULT_DEMO_DIM_GRID
    finetune_summary: str = ""
    skip_finetune: bool = False
    finetune_only: bool = False


def build_backbone(name: str) -> tuple[nn.Module, int]:
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
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim
    if name == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    if name == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier.in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    if name == "vit_b_16":
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        feat_dim = m.heads.head.in_features
        m.heads = nn.Identity()
        return m, feat_dim
    if name == "vgg16":
        m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[6].in_features
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
        return m, feat_dim
    raise ValueError(f"Unsupported model {name!r}. Choose from: {SUPPORTED_MODELS}")


class ImageOnlyERM(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor, demo: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        del demo
        return self.classifier(self.backbone(x))


class NaiveImageDemoERM(nn.Module):
    """Early fusion of image features + demographic embeddings (CheXpert ERM)."""

    def __init__(
        self,
        backbone_name: str,
        demo_cardinalities: dict[str, int],
        demo_embed_dim: int = 16,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name)
        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(max(v, 1), demo_embed_dim) for k, v in demo_cardinalities.items()}
        )
        in_dim = feat_dim + len(demo_cardinalities) * demo_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, demo: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if demo is None:
            raise ValueError("demo dictionary is required for NaiveImageDemoERM")
        z = self.backbone(x)
        demo_parts = []
        for attr, emb in self.embeddings.items():
            vals = demo[attr].long().clamp(min=0, max=emb.num_embeddings - 1)
            demo_parts.append(emb(vals))
        return self.classifier(torch.cat([z] + demo_parts, dim=1))


def amp_active(device: torch.device, use_amp: bool) -> bool:
    return bool(use_amp and device.type == "cuda")


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    if cfg.lr_backbone > 0 and cfg.lr_head > 0 and hasattr(model, "backbone"):
        backbone_params = list(model.backbone.parameters())
        head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.lr_backbone},
                {"params": head_params, "lr": cfg.lr_head},
            ],
            weight_decay=cfg.weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


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
def evaluate_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_demo_attrs: tuple[str, ...],
    selection_attr: str,
    use_amp: bool = False,
) -> dict[str, float]:
    """Val loss (5-path BCE) + overall/subgroup macro AUC on ``selection_attr``."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    logits_all, labels_all, groups_all = [], [], []
    loss_sum, n = 0.0, 0
    num_subgroups = ATTR_CARDINALITIES[selection_attr]
    amp_on = amp_active(device, use_amp)

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        g = batch[selection_attr].to(device, non_blocking=True).clamp(min=0)
        demo = {k: batch[k].to(device, non_blocking=True) for k in forward_demo_attrs}
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x, demo if forward_demo_attrs else None)
            eval_logits, eval_labels = eval_subset(logits, y)
            loss = criterion(eval_logits, eval_labels)
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


def selection_score(
    report: dict[str, float],
    *,
    use_demographics: bool,
    baseline_report: dict[str, float] | None,
    selection_attr: str,
) -> float:
    num_subgroups = ATTR_CARDINALITIES[selection_attr]
    if use_demographics:
        if baseline_report is None:
            raise ValueError("baseline_report required for min_improvement selection")
        return min_subgroup_macro_auc_improvement(baseline_report, report, num_subgroups)
    return float(report["worst_subgroup_macro_auc"])


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    forward_demo_attrs: tuple[str, ...],
    baseline_report: dict[str, float] | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    """Train with val checkpointing. Returns model + best-epoch val report."""
    optimizer = build_optimizer(model, cfg)
    criterion = nn.BCEWithLogitsLoss()
    amp_on = amp_active(device, cfg.use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_report: dict[str, float] = {}
    metric_name = "min_macro_auc_improvement" if cfg.use_demographics else "worst_subgroup_macro_auc"

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            demo = {k: batch[k].to(device, non_blocking=True) for k in forward_demo_attrs}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_on):
                logits = model(x, demo if forward_demo_attrs else None)
                loss = criterion(logits, y)  # full 13-pathology BCE
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_report = evaluate_report(
            model,
            val_loader,
            device,
            forward_demo_attrs,
            cfg.selection_attr,
            use_amp=cfg.use_amp,
        )
        score = selection_score(
            val_report,
            use_demographics=cfg.use_demographics,
            baseline_report=baseline_report,
            selection_attr=cfg.selection_attr,
        )
        val_report["selection_score"] = float(score) if not math.isnan(score) else float("nan")
        print(
            f"Epoch {epoch + 1:03d}: val_loss(5)={val_report['val_loss']:.4f} "
            f"macro_auc(5)={val_report['overall_macro_auc']:.4f} "
            f"worst_sub={val_report['worst_subgroup_macro_auc']:.4f} "
            f"{metric_name}={score:.4f}"
        )
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_report = dict(val_report)

    model.load_state_dict(best_state)
    if not best_report:
        best_report = evaluate_report(
            model, val_loader, device, forward_demo_attrs, cfg.selection_attr, use_amp=cfg.use_amp
        )
        best_report["selection_score"] = selection_score(
            best_report,
            use_demographics=cfg.use_demographics,
            baseline_report=baseline_report,
            selection_attr=cfg.selection_attr,
        )
    return model, best_report


@torch.no_grad()
def collect_test_rows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    seed: int,
    forward_demo_attrs: tuple[str, ...],
    use_amp: bool = False,
) -> list[dict[str, Any]]:
    """One row per (sample × eval pathology), matching CheXpert/ERM CSV schema."""
    model.eval()
    rows: list[dict[str, Any]] = []
    amp_on = amp_active(device, use_amp)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        demo = {k: batch[k].to(device, non_blocking=True) for k in forward_demo_attrs}
        with torch.amp.autocast("cuda", enabled=amp_on):
            logits = model(x, demo if forward_demo_attrs else None)
        probs = torch.sigmoid(logits.float())
        pred_cls = (probs >= 0.5).long()

        sample_ids = batch["sample_id"]
        rel_paths = batch["rel_path"]
        sexes = batch["sex"].tolist()
        ages = batch["age"].tolist()
        labels_cpu = labels.cpu()
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


def parse_models(s: str) -> tuple[str, ...]:
    if s.strip().lower() == "all":
        return SUPPORTED_MODELS
    models = tuple(x.strip().lower() for x in s.split(",") if x.strip())
    for m in models:
        if m not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model {m!r}")
    return models


def parse_seeds(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def generate_random_seeds(num_seeds: int) -> tuple[int, ...]:
    return tuple(random.randint(0, 2**31 - 1) for _ in range(num_seeds))


def parse_demo_attrs(s: str) -> tuple[str, ...]:
    attrs = tuple(canonicalize_attr(x) for x in s.split(",") if x.strip())
    for attr in attrs:
        if attr not in DEMO_ATTR_CHOICES:
            raise ValueError(f"demographic attrs must be from {DEMO_ATTR_CHOICES}")
    if not attrs:
        raise ValueError("at least one demographic attribute is required")
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for a in attrs:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return tuple(out)


def parse_float_grid(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def parse_int_grid(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def run_group_name(use_demographics: bool, demographic_attrs: tuple[str, ...]) -> str:
    if not use_demographics:
        return "ERM"
    return "ERM_" + "_".join(demographic_attrs)


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


def image_only_ckpt_path(cfg: Config, model_name: str, seed: int) -> Path:
    return Path(cfg.output_dir) / "checkpoints" / "ERM" / f"seed_{seed}" / f"{model_name}_image_only.pt"


def build_model(cfg: Config, model_name: str, device: torch.device) -> nn.Module:
    if cfg.use_demographics:
        demo_card = {a: ATTR_CARDINALITIES[a] for a in cfg.demographic_attrs}
        return NaiveImageDemoERM(
            model_name,
            demo_cardinalities=demo_card,
            demo_embed_dim=cfg.demo_embed_dim,
            num_classes=NUM_CLASSES,
        ).to(device)
    return ImageOnlyERM(model_name, num_classes=NUM_CLASSES).to(device)


def load_baseline_report(
    cfg: Config,
    model_name: str,
    seed: int,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Load image-only ERM checkpoint for ``seed`` and evaluate subgroup macro AUCs on val."""
    ckpt = image_only_ckpt_path(cfg, model_name, seed)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Image-only ERM checkpoint required for min_improvement baseline: {ckpt}\n"
            "Train image-only ERM first (same seeds)."
        )
    baseline = ImageOnlyERM(model_name, num_classes=NUM_CLASSES).to(device)
    baseline.load_state_dict(torch.load(ckpt, map_location=device))
    report = evaluate_report(
        baseline,
        val_loader,
        device,
        forward_demo_attrs=(),
        selection_attr=cfg.selection_attr,
        use_amp=cfg.use_amp,
    )
    print(
        f"Baseline ERM val: macro_auc={report['overall_macro_auc']:.4f} "
        f"worst_sub={report['worst_subgroup_macro_auc']:.4f} ({ckpt.name})"
    )
    return report


def apply_hparams_to_cfg(cfg: Config, hparams: dict[str, Any]) -> Config:
    cfg = copy.deepcopy(cfg)
    if "lr" in hparams:
        cfg.lr = float(hparams["lr"])
    if "weight_decay" in hparams:
        cfg.weight_decay = float(hparams["weight_decay"])
    if "batch_size" in hparams:
        cfg.batch_size = int(hparams["batch_size"])
    if "demo_embed_dim" in hparams:
        cfg.demo_embed_dim = int(hparams["demo_embed_dim"])
    if "epochs" in hparams:
        cfg.epochs = int(hparams["epochs"])
    if "lr_backbone" in hparams:
        cfg.lr_backbone = float(hparams["lr_backbone"])
    if "lr_head" in hparams:
        cfg.lr_head = float(hparams["lr_head"])
    return cfg


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    return build_frontal_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        pin_memory=True,
        aug_style="erm",
    )


def run_finetune(
    cfg: Config,
    model_name: str,
    finetune_seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Sequential hparam search on one seed; select by worst_subgroup or min_improvement."""
    forward_attrs = cfg.demographic_attrs if cfg.use_demographics else tuple()
    baseline_report = None
    if cfg.use_demographics:
        baseline_report = load_baseline_report(cfg, model_name, finetune_seed, val_loader, device)

    metric_name = "min_macro_auc_improvement" if cfg.use_demographics else "worst_subgroup_macro_auc"
    mid_wd = cfg.weight_decay_grid[len(cfg.weight_decay_grid) // 2]
    mid_bs = cfg.batch_size_grid[len(cfg.batch_size_grid) // 2] if cfg.batch_size_grid else cfg.batch_size
    best_hparams: dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": mid_wd,
        "batch_size": int(mid_bs),
        "demo_embed_dim": cfg.demo_embed_dim,
        "epochs": cfg.epochs,
    }
    trials: list[dict[str, Any]] = []
    loader_cache: dict[int, tuple[DataLoader, DataLoader]] = {
        int(cfg.batch_size): (train_loader, val_loader),
    }

    def _loaders_for(batch_size: int) -> tuple[DataLoader, DataLoader]:
        bs = int(batch_size)
        if bs not in loader_cache:
            tmp = copy.deepcopy(cfg)
            tmp.batch_size = bs
            tr, va, _te = build_loaders(tmp)
            loader_cache[bs] = (tr, va)
            print(f"Rebuilt dataloaders for batch_size={bs}")
        return loader_cache[bs]

    def _trial(hparams: dict[str, Any], stage: str) -> dict[str, Any]:
        trial_cfg = apply_hparams_to_cfg(cfg, hparams)
        set_seed(finetune_seed)
        model = build_model(trial_cfg, model_name, device)
        tr, va = _loaders_for(int(trial_cfg.batch_size))
        print(f"\n--- Finetune [{stage}] hparams={hparams} ---")
        model, report = train_one_run(
            model, tr, va, device, trial_cfg, forward_attrs, baseline_report
        )
        score = float(report.get("selection_score", float("nan")))
        entry = {"stage": stage, "hparams": dict(hparams), "metrics": report, "selection_score": score}
        trials.append(entry)
        return entry

    print(f"\n========== Finetune LR grid ({metric_name}) ==========")
    best_lr_score = float("-inf")
    best_lr = best_hparams["lr"]
    for lr in cfg.lr_grid:
        hp = {
            **best_hparams,
            "lr": float(lr),
            "weight_decay": float(mid_wd),
            "batch_size": int(mid_bs),
        }
        entry = _trial(hp, "lr")
        if not math.isnan(entry["selection_score"]) and entry["selection_score"] > best_lr_score:
            best_lr_score = entry["selection_score"]
            best_lr = float(lr)
            best_hparams = {**best_hparams, **entry["hparams"]}

    print(f"\n========== Finetune weight_decay grid (lr={best_lr}) ==========")
    best_wd_score = float("-inf")
    for wd in cfg.weight_decay_grid:
        hp = {
            **best_hparams,
            "lr": float(best_lr),
            "weight_decay": float(wd),
            "batch_size": int(mid_bs),
        }
        entry = _trial(hp, "weight_decay")
        if not math.isnan(entry["selection_score"]) and entry["selection_score"] > best_wd_score:
            best_wd_score = entry["selection_score"]
            best_hparams = {**best_hparams, **entry["hparams"]}

    if cfg.batch_size_grid:
        print(
            f"\n========== Finetune batch_size grid "
            f"(lr={best_hparams['lr']}, wd={best_hparams['weight_decay']}) =========="
        )
        best_bs_score = float("-inf")
        for bs in cfg.batch_size_grid:
            hp = {**best_hparams, "batch_size": int(bs)}
            entry = _trial(hp, "batch_size")
            if not math.isnan(entry["selection_score"]) and entry["selection_score"] > best_bs_score:
                best_bs_score = entry["selection_score"]
                best_hparams = {**best_hparams, **entry["hparams"]}

    if cfg.use_demographics and cfg.demo_embed_dim_grid:
        print("\n========== Finetune demo_embed_dim grid ==========")
        best_dim_score = float("-inf")
        for dim in cfg.demo_embed_dim_grid:
            hp = {**best_hparams, "demo_embed_dim": int(dim)}
            entry = _trial(hp, "demo_embed_dim")
            if not math.isnan(entry["selection_score"]) and entry["selection_score"] > best_dim_score:
                best_dim_score = entry["selection_score"]
                best_hparams = {**best_hparams, **entry["hparams"]}

    summary = {
        "run_group": run_group_name(cfg.use_demographics, cfg.demographic_attrs),
        "model": model_name,
        "finetune_seed": int(finetune_seed),
        "selection_metric": metric_name,
        "selection_attr": cfg.selection_attr,
        "use_demographics": cfg.use_demographics,
        "demographic_attrs": list(cfg.demographic_attrs),
        "train_findings": list(FINDING_COLS),
        "eval_findings": list(EVAL_FINDING_COLS),
        "epochs": cfg.epochs,
        "hparams": best_hparams,
        "grids": {
            "lr": list(cfg.lr_grid),
            "weight_decay": list(cfg.weight_decay_grid),
            "batch_size": list(cfg.batch_size_grid),
            "demo_embed_dim": list(cfg.demo_embed_dim_grid) if cfg.use_demographics else [],
        },
        "trials": [
            {
                "stage": t["stage"],
                "hparams": t["hparams"],
                "selection_score": t["selection_score"],
                "overall_macro_auc": t["metrics"].get("overall_macro_auc"),
                "worst_subgroup_macro_auc": t["metrics"].get("worst_subgroup_macro_auc"),
            }
            for t in trials
        ],
    }
    return summary


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def run_training(cfg: Config) -> None:
    if not cfg.seeds:
        raise ValueError("No seeds resolved")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_group = run_group_name(cfg.use_demographics, cfg.demographic_attrs)
    out_dir = Path(cfg.output_dir)
    ckpt_root = out_dir / "checkpoints" / run_group
    results_root = out_dir / "results" / run_group
    ckpt_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    if cfg.use_demographics and not cfg.demographic_attrs:
        raise ValueError("--demographic-attrs required with --use-demographics")
    if cfg.selection_attr not in DEMO_ATTR_CHOICES:
        raise ValueError(f"selection_attr must be sex|age; got {cfg.selection_attr!r}")

    metric_name = "min_macro_auc_improvement" if cfg.use_demographics else "worst_subgroup_macro_auc"
    print(f"Device: {device}")
    print(f"Run group: {run_group}")
    print(f"Models: {cfg.models}")
    print(f"Seeds: {cfg.seeds}")
    print(f"Epochs: {cfg.epochs} | batch_size={cfg.batch_size} | amp={cfg.use_amp}")
    print(f"Train findings: {NUM_CLASSES} | Eval findings: {len(EVAL_FINDING_COLS)}")
    print(f"Checkpoint metric: {metric_name} (attr={cfg.selection_attr})")
    print(f"Use demographics: {cfg.use_demographics}, attrs={cfg.demographic_attrs}")

    train_loader, val_loader, test_loader = build_loaders(cfg)
    forward_attrs = cfg.demographic_attrs if cfg.use_demographics else tuple()

    fields = [
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

    for seed in cfg.seeds:
        seed_results_dir = results_root / f"seed_{seed}"
        seed_results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = seed_results_dir / "test_predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    for model_name in cfg.models:
        train_cfg = copy.deepcopy(cfg)
        summary_path = (
            Path(cfg.finetune_summary)
            if cfg.finetune_summary
            else Path(cfg.output_dir) / "finetune" / f"finetune_summary_{run_group}__{model_name}.json"
        )

        if not train_cfg.skip_finetune:
            if summary_path.is_file() and not train_cfg.finetune_only:
                print(f"Finetune summary exists, loading: {summary_path}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                print(f"\n########## Finetune {model_name} on seed {cfg.seeds[0]} ##########")
                summary = run_finetune(
                    train_cfg, model_name, int(cfg.seeds[0]), train_loader, val_loader, device
                )
                save_json(summary_path, summary)
            train_cfg = apply_hparams_to_cfg(train_cfg, summary["hparams"])
            print(f"Using hparams: {summary['hparams']}")
            if int(train_cfg.batch_size) != int(cfg.batch_size):
                train_loader, val_loader, test_loader = build_loaders(train_cfg)
                print(f"Using batch_size={train_cfg.batch_size} from finetune summary")
        elif cfg.finetune_summary or summary_path.is_file():
            path = Path(cfg.finetune_summary) if cfg.finetune_summary else summary_path
            summary = json.loads(path.read_text(encoding="utf-8"))
            train_cfg = apply_hparams_to_cfg(train_cfg, summary["hparams"])
            print(f"Loaded hparams from {path}: {summary['hparams']}")
            if int(train_cfg.batch_size) != int(cfg.batch_size):
                train_loader, val_loader, test_loader = build_loaders(train_cfg)
                print(f"Using batch_size={train_cfg.batch_size} from finetune summary")
        else:
            print("skip_finetune=1 and no summary file; using CLI/default hparams")

        if train_cfg.finetune_only:
            print("finetune_only=1; skipping multi-seed training.")
            continue

        for seed in cfg.seeds:
            print("=" * 80)
            print(
                f"Train model={model_name} seed={seed} "
                f"lr={train_cfg.lr} wd={train_cfg.weight_decay} bs={train_cfg.batch_size} "
                f"demo_dim={train_cfg.demo_embed_dim} metric={metric_name}"
            )
            train_loader, val_loader, test_loader = build_loaders(train_cfg)
            set_seed(int(seed))
            model = build_model(train_cfg, model_name, device)
            baseline_report = None
            if train_cfg.use_demographics:
                baseline_report = load_baseline_report(
                    train_cfg, model_name, int(seed), val_loader, device
                )

            model, best_report = train_one_run(
                model, train_loader, val_loader, device, train_cfg, forward_attrs, baseline_report
            )

            seed_ckpt_dir = ckpt_root / f"seed_{seed}"
            seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
            run_name = (
                f"{model_name}_naive_demo" if train_cfg.use_demographics else f"{model_name}_image_only"
            )
            ckpt_path = seed_ckpt_dir / f"{run_name}.pt"
            torch.save(model.state_dict(), ckpt_path)
            meta_path = seed_ckpt_dir / f"{run_name}_val_metrics.json"
            save_json(
                meta_path,
                {
                    "seed": int(seed),
                    "selection_metric": metric_name,
                    "selection_attr": train_cfg.selection_attr,
                    "hparams": {
                        "lr": train_cfg.lr,
                        "weight_decay": train_cfg.weight_decay,
                        "batch_size": train_cfg.batch_size,
                        "demo_embed_dim": train_cfg.demo_embed_dim,
                        "epochs": train_cfg.epochs,
                    },
                    "val_metrics": best_report,
                },
            )
            print(f"Saved checkpoint: {ckpt_path}")

            csv_path = results_root / f"seed_{seed}" / "test_predictions.csv"
            rows = collect_test_rows(
                model,
                test_loader,
                device,
                run_name,
                int(seed),
                forward_attrs,
                use_amp=train_cfg.use_amp,
            )
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerows(rows)
            print(f"Updated test CSV: {csv_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CheXpert ERM with finetune → multi-seed training (worst-subgroup / min-improvement)."
    )
    p.add_argument("--data-root", type=str, default=Config.data_root)
    p.add_argument("--output-dir", type=str, default=Config.output_dir)
    p.add_argument("--models", type=str, default="resnet50")
    p.add_argument("--use-demographics", action="store_true")
    p.add_argument("--demographic-attrs", type=str, default="sex")
    p.add_argument(
        "--selection-attr",
        type=str,
        default="",
        help="Attr for worst_subgroup / min_improvement (default: first demo attr, else sex).",
    )
    p.add_argument("--seeds", type=str, default="random")
    p.add_argument("--num-random-seeds", type=int, default=5)
    p.add_argument("--seeds-file", type=str, default="")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--lr-backbone", type=float, default=0.0)
    p.add_argument("--lr-head", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument("--image-size", type=int, default=Config.image_size)
    p.add_argument("--demo-embed-dim", type=int, default=Config.demo_embed_dim)
    p.add_argument("--lr-grid", type=str, default="1e-4,5e-5,2e-5")
    p.add_argument("--weight-decay-grid", type=str, default="1e-4,1e-5,2e-5")
    p.add_argument("--batch-size-grid", type=str, default="32,64,128,256")
    p.add_argument("--demo-embed-dim-grid", type=str, default="8,16,32")
    p.add_argument("--finetune-summary", type=str, default="", help="Path to load/save finetune JSON.")
    p.add_argument("--skip-finetune", action="store_true", help="Skip search; use summary or CLI hparams.")
    p.add_argument("--finetune-only", action="store_true", help="Only run finetune on first seed.")
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--no-use-amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    seeds_file = Path(args.seeds_file) if str(args.seeds_file).strip() else (out_dir / "experiment_seeds.json")

    cfg = Config()
    cfg.data_root = args.data_root
    cfg.output_dir = str(out_dir)
    cfg.models = parse_models(args.models)
    cfg.use_demographics = bool(args.use_demographics)
    cfg.demographic_attrs = parse_demo_attrs(args.demographic_attrs) if cfg.use_demographics else tuple()
    if str(args.selection_attr).strip():
        cfg.selection_attr = canonicalize_attr(args.selection_attr)
    elif cfg.demographic_attrs:
        cfg.selection_attr = cfg.demographic_attrs[0]
    else:
        cfg.selection_attr = "sex"

    cfg.seeds = resolve_seeds(
        seeds_arg=args.seeds,
        num_random_seeds=args.num_random_seeds,
        seeds_file=seeds_file if seeds_file.is_file() else None,
        save_seeds_file=seeds_file,
    )
    cfg.epochs = int(args.epochs)
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.lr_backbone = args.lr_backbone
    cfg.lr_head = args.lr_head
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers
    cfg.image_size = args.image_size
    cfg.demo_embed_dim = args.demo_embed_dim
    cfg.lr_grid = parse_float_grid(args.lr_grid)
    cfg.weight_decay_grid = parse_float_grid(args.weight_decay_grid)
    cfg.batch_size_grid = parse_int_grid(args.batch_size_grid)
    cfg.demo_embed_dim_grid = parse_int_grid(args.demo_embed_dim_grid)
    cfg.finetune_summary = str(args.finetune_summary).strip()
    cfg.skip_finetune = bool(args.skip_finetune)
    cfg.finetune_only = bool(args.finetune_only)
    # CheXpert default AMP on; --no-use-amp forces off; --use-amp forces on.
    if args.no_use_amp:
        cfg.use_amp = False
    elif args.use_amp:
        cfg.use_amp = True
    # else keep Config.use_amp default (True)

    run_training(cfg)


if __name__ == "__main__":
    main()
