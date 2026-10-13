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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.utkface import (  # noqa: E402
    ATTR_CARDINALITIES,
    DEFAULT_DATA_ROOT,
    DEFAULT_TARGET,
    UTKFaceDataConfig,
    build_utkface_dataloaders,
)

SUPPORTED_MODELS: tuple[str, ...] = (
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "vit_b_16",
    "vgg16",
)
# Sensitive / demo attrs (prediction target defaults to gender).
DEMO_ATTR_CHOICES: tuple[str, ...] = ("age", "race")
TARGET_CHOICES: tuple[str, ...] = ("age", "gender", "race")

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


@dataclass
class Config:
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = "utkface_erm_outputs"
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    lr_backbone: float = 0.0
    lr_head: float = 0.0
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224
    models: tuple[str, ...] = ("vit_b_16",)
    seeds: tuple[int, ...] = ()
    target: str = DEFAULT_TARGET  # prediction label (default gender)
    use_demographics: bool = False
    demographic_attrs: tuple[str, ...] = ()
    # Attr used for worst-subgroup / min-improvement ACC (defaults to first demo attr or race).
    selection_attr: str = "race"
    use_amp: bool = False
    demo_embed_dim: int = 16
    # Finetune grids (sequential: lr → weight_decay → batch_size → [demo_embed_dim])
    lr_grid: tuple[float, ...] = DEFAULT_LR_GRID
    weight_decay_grid: tuple[float, ...] = DEFAULT_WD_GRID
    batch_size_grid: tuple[int, ...] = DEFAULT_BATCH_SIZE_GRID
    demo_embed_dim_grid: tuple[int, ...] = DEFAULT_DEMO_DIM_GRID
    finetune_summary: str = ""
    skip_finetune: bool = False
    finetune_only: bool = False

    @property
    def num_classes(self) -> int:
        return int(ATTR_CARDINALITIES[self.target])


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
    def __init__(self, backbone_name: str, num_classes: int = 2):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor, demo: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class NaiveImageDemoERM(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        demo_cardinalities: dict[str, int],
        demo_embed_dim: int = 16,
        num_classes: int = 2,
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


def demo_embedding_cardinality(attr: str) -> int:
    if attr not in ATTR_CARDINALITIES:
        raise ValueError(f"Unknown demographic attr {attr!r}")
    # Exact class count (age 0..2, race 0..4); no +1 offset.
    return int(ATTR_CARDINALITIES[attr])


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


def classification_report(
    probs: torch.Tensor, labels: torch.Tensor, subgroups: torch.Tensor, num_subgroups: int
) -> dict[str, float]:
    """Overall + per-subgroup ACC (primary) and AUC when binary / well-defined."""
    labels_np = labels.cpu().numpy()
    preds_np = probs.argmax(dim=1).cpu().numpy()
    subgroups_np = subgroups.cpu().numpy()
    report: dict[str, float] = {
        "overall_acc": float((preds_np == labels_np).mean()) if len(labels_np) else float("nan"),
    }

    # Binary AUC on class-1 score when applicable.
    if probs.size(1) == 2:
        scores_np = probs[:, 1].cpu().numpy()
        try:
            report["overall_auc"] = float(roc_auc_score(labels_np, scores_np))
        except Exception:
            report["overall_auc"] = float("nan")
    else:
        report["overall_auc"] = float("nan")
        scores_np = None

    for g in range(num_subgroups):
        mask = subgroups_np == g
        n = int(mask.sum())
        if n > 0:
            report[f"subgroup_{g}_acc"] = float((preds_np[mask] == labels_np[mask]).mean())
        else:
            report[f"subgroup_{g}_acc"] = float("nan")
        if scores_np is not None and n > 1 and len(set(labels_np[mask].tolist())) > 1:
            try:
                report[f"subgroup_{g}_auc"] = float(roc_auc_score(labels_np[mask], scores_np[mask]))
            except Exception:
                report[f"subgroup_{g}_auc"] = float("nan")
        else:
            report[f"subgroup_{g}_auc"] = float("nan")
    return report


def worst_subgroup_acc(report: dict[str, float], num_subgroups: int) -> float:
    vals = [float(report.get(f"subgroup_{g}_acc", float("nan"))) for g in range(num_subgroups)]
    vals = [v for v in vals if not math.isnan(v)]
    return min(vals) if vals else float("nan")


def min_subgroup_acc_improvement(
    base_report: dict[str, float], compare_report: dict[str, float], num_subgroups: int
) -> float:
    deltas = []
    for g in range(num_subgroups):
        b = float(base_report.get(f"subgroup_{g}_acc", float("nan")))
        c = float(compare_report.get(f"subgroup_{g}_acc", float("nan")))
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
    """Overall + per-subgroup ACC/AUC on ``selection_attr``, plus val_loss."""
    model.eval()
    logits_all, labels_all, groups_all = [], [], []
    loss_sum, n = 0.0, 0
    num_subgroups = ATTR_CARDINALITIES[selection_attr]

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        g = batch[selection_attr].to(device, non_blocking=True).clamp(min=0)
        demo = {k: batch[k].to(device, non_blocking=True) for k in forward_demo_attrs}
        with torch.cuda.amp.autocast(enabled=amp_active(device, use_amp)):
            logits = model(x, demo if forward_demo_attrs else None)
            loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs
        logits_all.append(logits.float().cpu())
        labels_all.append(y.cpu())
        groups_all.append(g.cpu())

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    groups = torch.cat(groups_all)
    probs = F.softmax(logits, dim=1)
    report = classification_report(probs, labels, groups, num_subgroups)
    report["val_loss"] = loss_sum / max(n, 1)
    report["worst_subgroup_acc"] = worst_subgroup_acc(report, num_subgroups)
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
        return min_subgroup_acc_improvement(baseline_report, report, num_subgroups)
    return float(report["worst_subgroup_acc"])


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
    scaler = torch.cuda.amp.GradScaler(enabled=amp_active(device, cfg.use_amp))
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_report: dict[str, float] = {}
    metric_name = "min_acc_improvement" if cfg.use_demographics else "worst_subgroup_acc"

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            demo = {k: batch[k].to(device, non_blocking=True) for k in forward_demo_attrs}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_active(device, cfg.use_amp)):
                logits = model(x, demo if forward_demo_attrs else None)
                loss = F.cross_entropy(logits, y)
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
            f"Epoch {epoch + 1:03d}: val_loss={val_report['val_loss']:.4f} "
            f"overall_acc={val_report['overall_acc']:.4f} "
            f"worst_sub_acc={val_report['worst_subgroup_acc']:.4f} "
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
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        demo = {k: batch[k].to(device, non_blocking=True) for k in forward_demo_attrs}
        with torch.cuda.amp.autocast(enabled=amp_active(device, use_amp)):
            logits = model(x, demo if forward_demo_attrs else None)
        logits = logits.float()
        probs = F.softmax(logits, dim=1)
        pred_cls = probs.argmax(dim=1)
        # Binary: positive-class score; multi-class: max probability.
        if probs.size(1) == 2:
            pred_prob = probs[:, 1]
            pos_logit = logits[:, 1]
        else:
            pred_prob = probs.max(dim=1).values
            pos_logit = logits.gather(1, pred_cls.unsqueeze(1)).squeeze(1)
        for i in range(len(batch["sample_id"])):
            rows.append(
                {
                    "sample_id": batch["sample_id"][i],
                    "true_label": int(y[i].item()),
                    "logit": float(pos_logit[i].item()),
                    "prediction_probability": float(pred_prob[i].item()),
                    "pred_class": int(pred_cls[i].item()),
                    "age": int(batch["age"][i].item()),
                    "gender": int(batch["gender"][i].item()),
                    "race": int(batch["race"][i].item()),
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
    attrs = tuple(x.strip().lower() for x in s.split(",") if x.strip())
    for attr in attrs:
        if attr not in DEMO_ATTR_CHOICES:
            raise ValueError(f"demographic attrs must be from {DEMO_ATTR_CHOICES}")
    return attrs


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


def default_finetune_summary_path(cfg: Config) -> Path:
    run_group = run_group_name(cfg.use_demographics, cfg.demographic_attrs)
    return Path(cfg.output_dir) / "finetune" / f"finetune_summary_{run_group}.json"


def image_only_ckpt_path(cfg: Config, model_name: str, seed: int) -> Path:
    return Path(cfg.output_dir) / "checkpoints" / "ERM" / f"seed_{seed}" / f"{model_name}_image_only.pt"


def build_model(cfg: Config, model_name: str, device: torch.device) -> nn.Module:
    n_cls = cfg.num_classes
    if cfg.use_demographics:
        demo_card = {a: demo_embedding_cardinality(a) for a in cfg.demographic_attrs}
        return NaiveImageDemoERM(
            model_name,
            demo_cardinalities=demo_card,
            demo_embed_dim=cfg.demo_embed_dim,
            num_classes=n_cls,
        ).to(device)
    return ImageOnlyERM(model_name, num_classes=n_cls).to(device)


def load_baseline_report(
    cfg: Config,
    model_name: str,
    seed: int,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Load image-only ERM checkpoint for ``seed`` and evaluate subgroup ACCs on val."""
    ckpt = image_only_ckpt_path(cfg, model_name, seed)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Image-only ERM checkpoint required for min_improvement baseline: {ckpt}\n"
            "Train image-only ERM first (same seeds)."
        )
    baseline = ImageOnlyERM(model_name, num_classes=cfg.num_classes).to(device)
    baseline.load_state_dict(torch.load(ckpt, map_location=device))
    report = evaluate_report(
        baseline, val_loader, device, forward_demo_attrs=(), selection_attr=cfg.selection_attr, use_amp=cfg.use_amp
    )
    print(
        f"Baseline ERM val: overall_acc={report['overall_acc']:.4f} "
        f"worst_sub_acc={report['worst_subgroup_acc']:.4f} ({ckpt.name})"
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
    data_cfg = UTKFaceDataConfig(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        pin_memory=True,
    )
    return build_utkface_dataloaders(cfg=data_cfg, target=cfg.target, demographic_mode=None)


def run_finetune(
    cfg: Config,
    model_name: str,
    finetune_seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Sequential hparam search on one seed; select by worst_subgroup_acc or min_acc_improvement."""
    forward_attrs = cfg.demographic_attrs if cfg.use_demographics else tuple()
    baseline_report = None
    if cfg.use_demographics:
        baseline_report = load_baseline_report(cfg, model_name, finetune_seed, val_loader, device)

    metric_name = "min_acc_improvement" if cfg.use_demographics else "worst_subgroup_acc"
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
    # Cache loaders by batch size so we only rebuild when needed.
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

    # 1) LR grid (wd + batch_size fixed at middle)
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

    # 2) Weight-decay grid (best lr, middle batch_size)
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

    # 3) Batch-size grid (best lr + wd)
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

    # 4) Demo embed dim (ERM+demo only)
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
        "target": cfg.target,
        "use_demographics": cfg.use_demographics,
        "demographic_attrs": list(cfg.demographic_attrs),
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
                "overall_acc": t["metrics"].get("overall_acc"),
                "worst_subgroup_acc": t["metrics"].get("worst_subgroup_acc"),
                "overall_auc": t["metrics"].get("overall_auc"),
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
    if cfg.target not in TARGET_CHOICES:
        raise ValueError(f"target must be one of {TARGET_CHOICES}; got {cfg.target!r}")
    if cfg.selection_attr not in DEMO_ATTR_CHOICES:
        raise ValueError(f"selection_attr must be age|race; got {cfg.selection_attr!r}")
    for attr in cfg.demographic_attrs:
        if attr == cfg.target:
            raise ValueError(f"demographic attr {attr!r} cannot equal prediction target {cfg.target!r}")

    metric_name = "min_acc_improvement" if cfg.use_demographics else "worst_subgroup_acc"
    print(f"Device: {device}")
    print(f"Run group: {run_group}")
    print(f"Target: {cfg.target} (num_classes={cfg.num_classes})")
    print(f"Models: {cfg.models}")
    print(f"Seeds: {cfg.seeds}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Checkpoint metric: {metric_name} (attr={cfg.selection_attr})")
    print(f"Use demographics: {cfg.use_demographics}, attrs={cfg.demographic_attrs}")

    train_loader, val_loader, test_loader = build_loaders(cfg)
    forward_attrs = cfg.demographic_attrs if cfg.use_demographics else tuple()

    fields = [
        "sample_id",
        "true_label",
        "logit",
        "prediction_probability",
        "pred_class",
        "age",
        "gender",
        "race",
        "model_name",
        "seed",
    ]

    # Truncate prediction CSVs once at the start of this run group.
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

        # ---- Finetune on first seed ----
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
            # Rebuild loaders if finetune selected a different batch size.
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

        # ---- Train all seeds with fixed hparams ----
        for seed in cfg.seeds:
            print("=" * 80)
            print(
                f"Train model={model_name} seed={seed} "
                f"lr={train_cfg.lr} wd={train_cfg.weight_decay} bs={train_cfg.batch_size} "
                f"demo_dim={train_cfg.demo_embed_dim} metric={metric_name}"
            )
            # Ensure loaders match selected batch size for this model.
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
    p = argparse.ArgumentParser(description="UTKFace ERM with finetune → multi-seed training (ACC selection).")
    p.add_argument("--data-root", type=str, default=Config.data_root)
    p.add_argument("--output-dir", type=str, default=Config.output_dir)
    p.add_argument("--models", type=str, default="vit_b_16")
    p.add_argument("--target", type=str, default=DEFAULT_TARGET, choices=list(TARGET_CHOICES))
    p.add_argument("--use-demographics", action="store_true")
    p.add_argument("--demographic-attrs", type=str, default="race")
    p.add_argument(
        "--selection-attr",
        type=str,
        default="",
        help="Attr for worst_subgroup_acc / min_acc_improvement (default: first demo attr, else race).",
    )
    p.add_argument("--seeds", type=str, default="random")
    p.add_argument("--num-random-seeds", type=int, default=5)
    p.add_argument("--seeds-file", type=str, default="")
    p.add_argument("--epochs", type=int, default=20)
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
    cfg.target = str(args.target).strip().lower()
    cfg.use_demographics = bool(args.use_demographics)
    cfg.demographic_attrs = parse_demo_attrs(args.demographic_attrs) if cfg.use_demographics else tuple()
    if str(args.selection_attr).strip():
        cfg.selection_attr = str(args.selection_attr).strip().lower()
    elif cfg.demographic_attrs:
        cfg.selection_attr = cfg.demographic_attrs[0]
    else:
        cfg.selection_attr = "race"

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
    if args.no_use_amp:
        cfg.use_amp = False
    elif args.use_amp:
        cfg.use_amp = True

    run_training(cfg)


if __name__ == "__main__":
    main()
