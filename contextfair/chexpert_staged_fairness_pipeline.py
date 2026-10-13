from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


DEFAULT_CHECKPOINT_DIR = "checkpoints/chexpert_contextfair"
SCRIPT_DIR = Path(__file__).resolve().parent

_COMP_MODELS_ROOT = Path(__file__).resolve().parent.parent
if str(_COMP_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMP_MODELS_ROOT))

from datasets.chexpert import (  # noqa: E402
    AGE_SUBGROUP_NAMES,
    DEFAULT_DATA_ROOT,
    EVAL_CLASS_INDICES,
    EVAL_FINDING_COLS,
    NUM_CLASSES as NUM_ALL_FINDINGS,
    NUM_EVAL_CLASSES,
    SEX_SUBGROUP_NAMES,
    build_chexpert_dataloaders,
)

EVAL_DISEASE_NAMES: tuple[str, ...] = tuple(EVAL_FINDING_COLS)
TRAIN_DISEASE_CHOICES: tuple[int, ...] = (5, 13)
DEFAULT_TRAIN_DISEASES = 13

DEFAULT_ERM_HPARAMS_DIR = str(_COMP_MODELS_ROOT / "ERM" / "chexpert_outputs" / "finetune")
ERM_HPARAM_KEYS: tuple[str, ...] = ("lr", "weight_decay", "batch_size", "epochs")


def resolve_train_diseases(train_diseases: int | str) -> int:
    n = int(train_diseases)
    if n not in TRAIN_DISEASE_CHOICES:
        raise ValueError(f"train_diseases must be one of {TRAIN_DISEASE_CHOICES}, got {train_diseases!r}")
    return n


def num_classes_for_train_diseases(train_diseases: int | str) -> int:
    n = resolve_train_diseases(train_diseases)
    return NUM_EVAL_CLASSES if n == 5 else NUM_ALL_FINDINGS


def slice_eval_labels(labels: torch.Tensor) -> torch.Tensor:
    """Select the 5 competition disease columns from a 13-d tensor (labels/logits/probs)."""
    idx = torch.tensor(EVAL_CLASS_INDICES, device=labels.device, dtype=torch.long)
    return labels.index_select(-1, idx)


def to_eval_space(
    logits_or_probs: torch.Tensor, labels: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map model outputs (5 or 13) to the 5-disease eval space; slice labels likewise."""
    if logits_or_probs.shape[-1] == NUM_ALL_FINDINGS:
        out = slice_eval_labels(logits_or_probs)
    elif logits_or_probs.shape[-1] == NUM_EVAL_CLASSES:
        out = logits_or_probs
    else:
        raise ValueError(
            f"Expected last dim {NUM_EVAL_CLASSES} or {NUM_ALL_FINDINGS}, got {logits_or_probs.shape[-1]}"
        )
    if labels is None:
        return out, None
    if labels.shape[-1] == NUM_ALL_FINDINGS:
        y = slice_eval_labels(labels)
    elif labels.shape[-1] == NUM_EVAL_CLASSES:
        y = labels
    else:
        raise ValueError(f"Unexpected labels last dim {labels.shape[-1]}")
    return out, y


def bce_logits_loss(logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float(), reduction=reduction)


def bce_logits_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean BCE over diseases → one loss per sample [B]."""
    return F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none").mean(dim=1)


def kd_loss_multilabel(proposed_logits: torch.Tensor, baseline_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Eq. (17): L_kd — temperature-scaled soft-BCE KD from frozen Stage-1 baseline."""
    t = temperature
    p = torch.sigmoid(proposed_logits / t)
    q = torch.sigmoid(baseline_logits.detach() / t)
    eps = 1e-8
    loss = -(q * torch.log(p + eps) + (1.0 - q) * torch.log(1.0 - p + eps))
    return loss.mean() * (t * t)


def _per_disease_aucs(y: np.ndarray, p: np.ndarray) -> list[float]:
    aucs: list[float] = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y[:, j], p[:, j])))
        except Exception:
            continue
    return aucs


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = str(SCRIPT_DIR)
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    run_name: str = "default"

    demographic_mode: str = "sex"
    demographic_key: str = "sex"
    subgroup_names: tuple[str, ...] = SEX_SUBGROUP_NAMES
    num_subgroups: int = 2

    image_size: int = 224
    image_channels: int = 3
    train_diseases: int = DEFAULT_TRAIN_DISEASES  # 13 (match ERM/fairness) or 5; eval always 5
    num_classes: int = DEFAULT_TRAIN_DISEASES  # synced from train_diseases via apply_train_diseases_defaults
    feature_dim: int = 512
    demo_embed_dim: int = 32
    gamma: float = 0.5  # λ_film in Eq. (10)
    fusion_type: str = "film"  # film | scalar  (FiLM ≡ Eqs. 9–10)
    residual_blocks: int = 1
    head_layernorm: bool = True
    head_hidden_dim: int = 256
    head_dropout: float = 0.3
    pretrained: bool = True
    backbone: str = "resnet50"  # resnet50 | vit_b_16

    batch_size: int = 32
    num_workers: int = 8
    weight_decay: float = 1e-4
    stage1_epochs: int = 10
    stage2_epochs: int = 10
    stage4_epochs: int = 15
    stage1_lr: float = 1e-4
    stage2_lr: float = 1e-4
    stage4_head_lr: float = 1e-4
    stage4_encoder_lr: float = 2e-5

    stage4_unfreeze_encoder: bool = True
    stage4_unfreeze_from_block: int = 4  # ResNet50 stages 1-4; 4 = layer4 only (cheapest safe FT)
    stage4_unfreeze_include_fc: bool = True
    use_cosine_schedule_stage1: bool = False
    use_cosine_schedule_stage4: bool = True
    warmup_epochs_stage4: int = 2
    stage4_checkpoint_skip_epochs: int = 2
    min_lr_ratio: float = 0.01

    alpha: float = 50.0  # Eq. (8): σ(α d)
    lambda_cv: float = 0.5  # Eq. (7): P_cv = exp(−λ_cv · CV)
    eps: float = 1e-8
    lambda_imp: float = 0.2  # Eq. (15) L_imp = Var_g(Δ^L_g); λ_imp in Eq. (18)
    lambda_worst: float = 0.1  # Eq. (16) L_worst = −min_g(Δ^L_g); λ_worst in Eq. (18)
    lambda_kd: float = 0.5  # Eq. (17) L_kd; λ_kd in Eq. (18)
    temperature: float = 3.0
    fairness_use_relu: bool = False

    selection_metric_stage1: str = "worst_subgroup"
    selection_metric_stage2: str = "min_improvement"
    selection_metric_stage4: str = "min_improvement"
    use_overall_tie_break: bool = True

    erm_hparams_dir: str = DEFAULT_ERM_HPARAMS_DIR
    skip_erm_hparams: bool = False

    baseline_only: bool = False
    skip_baseline: bool = False

    seeds_arg: str = "random"
    num_random_seeds: int = 5
    seeds_file: str = ""

    seed: int = 42
    device: str = "auto"
    finetune_stages: bool = False
    finetune_sequential_lambdas: bool = False
    finetune_only_stage: str = "all"  # all | stage1 | stage2 | stage3 | stage4
    finetune_from_summary: str = ""
    save_finetune_checkpoints: bool = True
    apply_finetune_summary: str = ""
    stop_after_stage3: bool = False
    stage4_only: bool = False
    init_from_run_name: str = ""  # base run_name (fusion suffix optional) for --stage4-only

    stage1_lr_grid: tuple[float, ...] = (1e-4, 5e-5, 2e-5)
    stage1_weight_decay_grid: tuple[float, ...] = (1e-4, 1e-5)
    stage1_batch_size_grid: tuple[int, ...] = (32, 64, 128)
    stage2_demo_embed_dim_grid: tuple[int, ...] = (16, 32, 64)
    stage3_lambda_cv_grid: tuple[float, ...] = (0.5, 1.0)
    stage3_alpha_grid: tuple[float, ...] = (25.0, 50.0, 100.0)
    stage4_lambda_imp_grid: tuple[float, ...] = (0.1, 0.2, 0.5)
    stage4_lambda_worst_grid: tuple[float, ...] = (0.05, 0.1, 0.2)
    stage4_lambda_kd_grid: tuple[float, ...] = (0.3, 0.5)
    stage4_temperature_grid: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0)
    stage4_gamma_grid: tuple[float, ...] = (0.5, 1.0)
    stage4_head_lr_grid: tuple[float, ...] = (1e-4, 5e-5)
    stage4_encoder_lr_grid: tuple[float, ...] = (2e-5, 1e-5)
    stage4_epochs_grid: tuple[int, ...] = (10, 15)  # unused in sequential tune; fixed stage4_epochs
    stage4_unfreeze_from_block_grid: tuple[int, ...] = (2, 3, 4)  # ResNet50 stages 1–4 only


STAGE4_INT_HPARAMS: frozenset[str] = frozenset({"stage4_unfreeze_from_block"})
STAGE4_SEQUENTIAL_TUNING: tuple[tuple[str, str], ...] = (
    ("lambda_imp", "stage4_lambda_imp_grid"),
    ("lambda_worst", "stage4_lambda_worst_grid"),
    ("lambda_kd", "stage4_lambda_kd_grid"),
    ("temperature", "stage4_temperature_grid"),
    ("gamma", "stage4_gamma_grid"),
    ("stage4_head_lr", "stage4_head_lr_grid"),
    ("stage4_encoder_lr", "stage4_encoder_lr_grid"),
    ("stage4_unfreeze_from_block", "stage4_unfreeze_from_block_grid"),
)

SELECTION_METRICS: tuple[str, ...] = ("worst_subgroup", "overall_auc", "min_improvement")
STAGE4_CHECKPOINT_METRICS: tuple[str, ...] = ("min_improvement",)
FINETUNE_ONLY_STAGES: tuple[str, ...] = ("all", "stage1", "stage2", "stage3", "stage4")


def parse_float_grid(value: str) -> tuple[float, ...]:
    vals = tuple(float(v.strip()) for v in value.split(",") if v.strip())
    if not vals:
        raise ValueError("Grid argument must contain at least one float value.")
    return vals


def parse_int_grid(value: str) -> tuple[int, ...]:
    vals = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not vals:
        raise ValueError("Grid argument must contain at least one integer value.")
    return vals


def grid_middle(values: tuple[float, ...] | tuple[int, ...]) -> float:
    """Middle grid value: median element (odd length) or mean of two central elements (even length)."""
    if not values:
        raise ValueError("grid_middle requires a non-empty grid.")
    nums = sorted(float(v) for v in values)
    n = len(nums)
    mid = n // 2
    if n % 2 == 1:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def grid_middle_int(values: tuple[int, ...]) -> int:
    return int(round(grid_middle(values)))


def _stage4_grid_middle(cfg: Config, param_name: str, grid_attr: str) -> float | int:
    grid = getattr(cfg, grid_attr)
    if param_name in STAGE4_INT_HPARAMS:
        return grid_middle_int(grid)
    return float(grid_middle(grid))


def _initial_stage4_hparams(cfg: Config, *, use_grid_middles: bool) -> dict[str, Any]:
    if use_grid_middles:
        hparams = {
            param: _stage4_grid_middle(cfg, param, grid_attr)
            for param, grid_attr in STAGE4_SEQUENTIAL_TUNING
        }
    else:
        hparams = {
            "lambda_imp": float(cfg.lambda_imp),
            "lambda_worst": float(cfg.lambda_worst),
            "lambda_kd": float(cfg.lambda_kd),
            "temperature": float(cfg.temperature),
            "gamma": float(cfg.gamma),
            "stage4_head_lr": float(cfg.stage4_head_lr),
            "stage4_encoder_lr": float(cfg.stage4_encoder_lr),
            "stage4_unfreeze_from_block": int(cfg.stage4_unfreeze_from_block),
        }
    hparams["stage4_epochs"] = int(cfg.stage4_epochs)
    return hparams


def _stage4_summary_key(param_name: str) -> str:
    if param_name.startswith("lambda_"):
        return f"stage4_{param_name}"
    return f"stage4_{param_name}"


def apply_demographic_mode_defaults(cfg: Config) -> Config:
    mode = str(cfg.demographic_mode).strip().lower()
    if mode == "sex":
        cfg.demographic_key = "sex"
        cfg.num_subgroups = 2
        cfg.subgroup_names = SEX_SUBGROUP_NAMES
    elif mode == "age":
        cfg.demographic_key = "age"
        cfg.num_subgroups = 2
        cfg.subgroup_names = AGE_SUBGROUP_NAMES
    elif mode == "custom":
        pass
    else:
        raise ValueError("demographic_mode must be one of: sex, age, custom")
    return cfg


def apply_train_diseases_defaults(cfg: Config) -> Config:
    cfg.train_diseases = resolve_train_diseases(cfg.train_diseases)
    cfg.num_classes = num_classes_for_train_diseases(cfg.train_diseases)
    return cfg


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
    """Resolve a shared seed list, writing/reading ``{output_dir}/experiment_seeds.json``.

    ``seeds_arg="random"`` loads ``seeds_file``/``save_seeds_file`` if present, else
    generates ``num_random_seeds`` fresh seeds. Any explicit comma-separated
    ``seeds_arg`` is used as-is. The resolved seeds are (re)written to
    ``save_seeds_file`` (or ``seeds_file``) when provided, so subsequent calls
    (other demographic modes, multi-seed runs, etc.) reuse the same seeds.
    """
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
            print(f"Loaded shared seeds from {load_path}: {seeds}", file=sys.stderr)
        else:
            seeds = generate_random_seeds(num_random_seeds)
            print(f"Generated random seeds: {seeds}", file=sys.stderr)

    out_path = save_seeds_file or seeds_file
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"seeds": list(seeds), "num_seeds": len(seeds)}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote shared seeds to {out_path}", file=sys.stderr)
    return seeds


def erm_hparams_summary_path(erm_hparams_dir: str, backbone: str = "resnet50") -> Path:
    return Path(erm_hparams_dir) / f"finetune_summary_ERM__{backbone}.json"


def load_erm_stage1_hparams(cfg: Config, backbone: str = "resnet50") -> dict[str, Any] | None:
    """Load Stage-1 (image-only ERM) hparams from a Comp_models ERM finetune summary.

    Returns ``None`` when disabled (``skip_erm_hparams``), no ``erm_hparams_dir`` is
    set, or the summary file is missing/unusable — callers should fall back to the
    normal Stage-1 lr/weight_decay grid search in that case.
    """
    if cfg.skip_erm_hparams or not str(cfg.erm_hparams_dir).strip():
        return None
    path = erm_hparams_summary_path(cfg.erm_hparams_dir, backbone)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    hparams = payload.get("hparams")
    if not isinstance(hparams, dict):
        return None
    filtered = {k: hparams[k] for k in ERM_HPARAM_KEYS if k in hparams}
    if not filtered:
        return None
    print(
        f"[Stage1] Using ERM finetune hparams from {path}: {filtered} "
        "(skipping Stage-1 lr/weight_decay grid search; still training Stage-1 "
        "locally on this run's seeds with worst_subgroup selection)"
    )
    return filtered


def apply_erm_hparams_to_cfg(cfg: Config, hparams: dict[str, Any]) -> None:
    if "lr" in hparams:
        cfg.stage1_lr = float(hparams["lr"])
    if "weight_decay" in hparams:
        cfg.weight_decay = float(hparams["weight_decay"])
    if "batch_size" in hparams:
        cfg.batch_size = int(hparams["batch_size"])
    if "epochs" in hparams:
        cfg.stage1_epochs = int(hparams["epochs"])


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Shared CheXpert loaders from ContextFair_and_Comp_models/datasets.chexpert.

    Batches include a ``subgroup`` key for sex/age demographic modes (required by
    Stages 1–4 below), plus ``labels`` (train width: 5 or 13) and ``labels_eval``
    (always 5). ``demographic_mode="custom"`` is not supported by the shared loader
    (Comp_models only defines sex/age demographics for CheXpert).
    """
    demographic_mode = cfg.demographic_mode if cfg.demographic_mode in ("sex", "age") else ""
    if not demographic_mode:
        raise ValueError(
            f"demographic_mode={cfg.demographic_mode!r} is not supported by the shared "
            "ContextFair_and_Comp_models/datasets.chexpert loader; use 'sex' or 'age'."
        )
    return build_chexpert_dataloaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        demographic_mode=demographic_mode,
        num_subgroups=cfg.num_subgroups,
        train_diseases=cfg.train_diseases,
        pin_memory=True,
        aug_style="contextfair",
    )


class ResNet50Encoder(nn.Module):
    """ResNet-50 trunk + linear projection to feature_dim."""

    def __init__(self, feature_dim: int = 512, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        cache_dir = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_dir)

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                    if old_conv.bias is not None and backbone.conv1.bias is not None:
                        backbone.conv1.bias.copy_(old_conv.bias)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


FUSION_TYPES: tuple[str, ...] = ("film", "scalar")
SUPPORTED_BACKBONES: tuple[str, ...] = ("resnet50", "vit_b_16")


def normalize_fusion_type(fusion_type: str) -> str:
    normalized = str(fusion_type).strip().lower()
    if normalized not in FUSION_TYPES:
        raise ValueError(f"fusion_type must be one of {FUSION_TYPES}; got {fusion_type!r}")
    return normalized


def effective_run_name(run_name: str, fusion_type: str) -> str:
    fusion = normalize_fusion_type(fusion_type)
    suffix = f"__fusion_{fusion}"
    base = str(run_name).strip().rstrip("/")
    for existing in FUSION_TYPES:
        existing_suffix = f"__fusion_{existing}"
        if base.endswith(existing_suffix):
            base = base[: -len(existing_suffix)]
            break
    return f"{base}{suffix}"


def apply_effective_run_name(cfg: Config) -> None:
    cfg.run_name = effective_run_name(cfg.run_name, cfg.fusion_type)


def cfg_for_run_name(cfg: Config, run_name: str) -> Config:
    """Copy cfg with run_name rewritten to the effective fusion-suffixed name."""
    other = copy.deepcopy(cfg)
    other.run_name = effective_run_name(run_name, cfg.fusion_type)
    return other


def stage123_summary_path(cfg: Config) -> Path:
    return results_run_dir(cfg) / "stage123_summary.json"


def save_stage123_summary(
    cfg: Config,
    usefulness_score: float,
    usefulness_meta: dict[str, Any],
    baseline_val: dict[str, Any],
    demo_val: dict[str, Any],
) -> Path:
    out_dir = results_run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": cfg.run_name,
        "demographic_mode": cfg.demographic_mode,
        "seed": cfg.seed,
        "fusion_type": cfg.fusion_type,
        "usefulness_score": float(usefulness_score),
        "usefulness_meta": to_jsonable(usefulness_meta),
        "baseline_val": to_jsonable(baseline_val),
        "demo_val": to_jsonable(demo_val),
        "hparams": {
            "stage1_lr": cfg.stage1_lr,
            "stage2_lr": cfg.stage2_lr,
            "weight_decay": cfg.weight_decay,
            "demo_embed_dim": cfg.demo_embed_dim,
            "alpha": cfg.alpha,
            "lambda_cv": cfg.lambda_cv,
            "gamma": cfg.gamma,
        },
        "checkpoint_dirs": {
            "stage1": str(scratch_checkpoint_dir(cfg, "stage1")),
            "stage1_baseline_ref": str(scratch_checkpoint_dir(cfg, "stage1_baseline_ref")),
            "stage2": str(scratch_checkpoint_dir(cfg, "stage2")),
        },
    }
    path = stage123_summary_path(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    print(f"Saved Stages 1–3 summary → {path.resolve()}")
    return path


def load_stage123_summary(cfg: Config) -> dict[str, Any] | None:
    path = stage123_summary_path(cfg)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_stage123_for_stage4(
    cfg: Config,
    init_cfg: Config,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[ProposedFairnessModel, Stage1BaselineReference, float, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load Stage-1 ref + Stage-2 weights from init_cfg; usefulness from summary or recompute."""
    stage1_ref = Stage1BaselineReference.from_checkpoint(init_cfg, device)
    stage2_state = load_stage2_state_from_checkpoint(init_cfg)
    model = build_model(cfg, device)
    load_matching_state_dict(model, stage2_state)

    summary = load_stage123_summary(init_cfg)
    baseline_val: dict[str, Any]
    demo_val: dict[str, Any]
    if summary is not None and "usefulness_score" in summary:
        usefulness_score = float(summary["usefulness_score"])
        usefulness_meta = dict(summary.get("usefulness_meta") or {})
        baseline_val = dict(summary.get("baseline_val") or {})
        demo_val = dict(summary.get("demo_val") or {})
        print(
            f"Loaded usefulness_score={usefulness_score:.6f} from "
            f"{stage123_summary_path(init_cfg)}"
        )
    else:
        print("[WARN] stage123_summary.json missing usefulness; recomputing on val.")
        usefulness_score, usefulness_meta, baseline_val, demo_val = stage3_compute_usefulness(
            model, val_loader, device, cfg, stage1_ref
        )
    return model, stage1_ref, usefulness_score, usefulness_meta, baseline_val, demo_val


class ScalarAttentionGate(nn.Module):
    def __init__(self, feature_dim: int, demo_embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + demo_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([z, e], dim=1)))


class DemoFiLMGate(nn.Module):
    """
    FiLM-style gate for demographic embeddings (paper §4.4).

    Eq. (9):  [γ, β] = MLP_film([z̄ ; ē])
    """

    def __init__(self, feature_dim: int, demo_embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + demo_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, demo_embed_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq. (9): [γ, β] = MLP_film([z̄ ; ē])
        gamma, beta = self.net(torch.cat([z, e], dim=1)).chunk(2, dim=1)
        return gamma, beta


def build_demo_gate(fusion_type: str, feature_dim: int, demo_embed_dim: int) -> nn.Module:
    fusion_type = normalize_fusion_type(fusion_type)
    if fusion_type == "film":
        return DemoFiLMGate(feature_dim, demo_embed_dim)
    return ScalarAttentionGate(feature_dim, demo_embed_dim)


class LegacyFusionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim) if use_layernorm else None
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3, use_layernorm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layernorm else None
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm(h) if self.norm is not None else h
        return self.mlp(x)


class ResidualFusionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_blocks: int,
        dropout: float = 0.3,
        use_layernorm: bool = True,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("ResidualFusionHead requires num_blocks >= 1")

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, dropout=dropout, use_layernorm=use_layernorm)
            for _ in range(num_blocks)
        )
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)


def build_fusion_head(
    in_dim: int,
    num_classes: int,
    residual_blocks: int = 0,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    use_layernorm: bool = False,
) -> nn.Module:
    if residual_blocks <= 0:
        if use_layernorm:
            return LegacyFusionHead(in_dim, hidden_dim, num_classes, dropout, use_layernorm=True)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    return ResidualFusionHead(
        in_dim, hidden_dim, num_classes, residual_blocks, dropout, use_layernorm=use_layernorm
    )


class ProposedFairnessModel(nn.Module):
    """
    Shared ResNet-50 encoder with:
      - baseline head (image only), multi-label (13 or 5 findings)
      - demo head on [z; e]
      - proposed / ContextFair head implementing paper §4.4:
          Eq. (9)–(11) utility-guided FiLM (or scalar) modulation of ē
          Eq. (12) ŷ_cf = C_cf(F(z̄, ẽ)) with F = concat
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        feature_dim: int,
        num_subgroups: int,
        demo_embed_dim: int,
        num_classes: int,
        gamma: float,
        fusion_type: str,
        residual_blocks: int,
        head_layernorm: bool,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.gamma = gamma  # λ_film in Eq. (10)
        self.fusion_type = normalize_fusion_type(fusion_type)
        self.residual_blocks = max(0, int(residual_blocks))
        self.head_layernorm = bool(head_layernorm)

        # Learnable demographic embedding ē (paper §4.4)
        self.demo_embedding = nn.Embedding(num_subgroups, demo_embed_dim)
        fused_dim = feature_dim + demo_embed_dim

        self.baseline_head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )
        self.demo_head = build_fusion_head(
            fused_dim, num_classes, self.residual_blocks, head_hidden_dim, head_dropout, self.head_layernorm
        )
        self.demo_gate = build_demo_gate(self.fusion_type, feature_dim, demo_embed_dim)
        self.proposed_head = build_fusion_head(
            fused_dim, num_classes, self.residual_blocks, head_hidden_dim, head_dropout, self.head_layernorm
        )

    def _modulate_demographics(self, z: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_type == "film":
            # Eq. (9): [γ, β] = MLP_film([z̄ ; ē])
            film_gamma, film_beta = self.demo_gate(z, e)
            # Eq. (10): ē_mod = (1 + λ_film · γ) ⊙ ē + β   (self.gamma ≡ λ_film)
            e_mod = (1.0 + self.gamma * film_gamma) * e + film_beta
            return e_mod, film_gamma
        gate = self.demo_gate(z, e)
        e_mod = (1.0 + self.gamma * gate) * e
        return e_mod, gate

    def forward(
        self,
        x: torch.Tensor,
        subgroup: torch.Tensor,
        usefulness_score: float | torch.Tensor = 0.0,
    ) -> dict[str, torch.Tensor]:
        # z̄ = E_cf(x); ē = Embedding(subgroup)
        z = self.image_encoder(x)
        e = self.demo_embedding(subgroup)

        logits_baseline = self.baseline_head(z)
        logits_demo = self.demo_head(torch.cat([z, e], dim=1))

        if not torch.is_tensor(usefulness_score):
            usefulness_score = torch.tensor(usefulness_score, device=x.device, dtype=z.dtype)
        usefulness_score = usefulness_score.to(device=x.device, dtype=z.dtype)
        if usefulness_score.dim() == 0:
            usefulness_score = usefulness_score.view(1, 1).expand(x.size(0), 1)
        elif usefulness_score.dim() == 1:
            usefulness_score = usefulness_score.unsqueeze(1)

        # Eqs. (9)–(10) → ē_mod; Eq. (11): ẽ = u · ē_mod
        e_mod, gate = self._modulate_demographics(z, e)
        e_tilde = usefulness_score * e_mod
        # Eq. (12): ŷ_cf = C_cf(F(z̄, ẽ)) with F(z̄, ẽ) = [z̄ ; ẽ]
        logits_proposed = self.proposed_head(torch.cat([z, e_tilde], dim=1))

        return {
            "z": z,
            "e": e,
            "gate": gate,
            "e_tilde": e_tilde,
            "baseline": logits_baseline,
            "demo": logits_demo,
            "proposed": logits_proposed,
        }


FilmResidualFairnessModel = ProposedFairnessModel


class Stage1BaselineReference(nn.Module):
    """Frozen Stage-1 image encoder + baseline head for usefulness and Stage-4 loss terms."""

    def __init__(self, image_encoder: nn.Module, baseline_head: nn.Module):
        super().__init__()
        self.image_encoder = copy.deepcopy(image_encoder)
        self.baseline_head = copy.deepcopy(baseline_head)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @classmethod
    def from_model(cls, model: ProposedFairnessModel) -> Stage1BaselineReference:
        return cls(model.image_encoder, model.baseline_head)

    @classmethod
    def from_checkpoint(cls, cfg: Config, device: torch.device) -> Stage1BaselineReference:
        path = scratch_checkpoint_dir(cfg, "stage1_baseline_ref") / "checkpoint.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Stage-1 baseline reference not found: {path}. "
                "Run Stages 1–2 finetune first with checkpoint saving enabled."
            )
        payload = torch.load(path, map_location=device)
        probe = build_model(cfg, device)
        encoder = build_image_encoder(cfg).to(device)
        encoder.load_state_dict(payload["encoder_state"])
        baseline_head = copy.deepcopy(probe.baseline_head)
        baseline_head.load_state_dict(payload["baseline_head_state"])
        return cls(encoder, baseline_head).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.baseline_head(self.image_encoder(x))

    @torch.no_grad()
    def collect_predictions(
        self, dataloader: DataLoader, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sigmoid probs/labels in 5-disease eval space (for macro AUC)."""
        self.eval()
        all_probs, all_labels, all_groups = [], [], []
        for batch in dataloader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch.get("labels_eval", batch["labels"]).to(device, non_blocking=True)
            g = batch["subgroup"].to(device, non_blocking=True)
            logits = self(x)
            probs_e, y_e = to_eval_space(torch.sigmoid(logits), y)
            all_probs.append(probs_e.cpu())
            all_labels.append(y_e.cpu())
            all_groups.append(g.cpu())
        return torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_groups)

    @torch.no_grad()
    def auc_report(self, dataloader: DataLoader, device: torch.device, num_subgroups: int) -> dict[str, float]:
        probs, labels, groups = self.collect_predictions(dataloader, device)
        return binary_auc_report(probs, labels, groups, num_subgroups)

    @torch.no_grad()
    def compute_subgroup_losses(
        self, train_loader: DataLoader, device: torch.device, num_subgroups: int
    ) -> torch.Tensor:
        """Per-subgroup mean-BCE losses of the frozen Stage-1 baseline on the train split."""
        self.eval()
        loss_sum = torch.zeros(num_subgroups, device=device)
        count = torch.zeros(num_subgroups, device=device)
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)  # train width (5 or 13)
            g = batch["subgroup"].to(device, non_blocking=True)
            losses = bce_logits_per_sample(self(x), y)
            for group_id in range(num_subgroups):
                mask = g == group_id
                if mask.sum() > 0:
                    loss_sum[group_id] += losses[mask].sum()
                    count[group_id] += mask.sum()
        return (loss_sum / count.clamp(min=1)).detach()


RESNET50_NUM_STAGES = 4  # layer1..layer4


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


class ViTB16Encoder(nn.Module):
    """ViT-B/16 encoder → feature_dim projection (CLS token)."""

    def __init__(self, feature_dim: int = 512, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        cache_dir = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_dir)

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vit_b_16(weights=weights)
        if in_channels != 3:
            raise ValueError("ViT-B/16 ContextFair encoder currently requires in_channels=3")
        hidden = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.fc = nn.Linear(hidden, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


VIT_B16_NUM_BLOCKS = 12


def freeze_vit_encoder(encoder: ViTB16Encoder) -> None:
    set_requires_grad(encoder, False)


def unfreeze_vit_encoder_tail(
    encoder: ViTB16Encoder,
    from_block: int = 8,
    include_fc: bool = True,
) -> int:
    """Unfreeze ViT transformer blocks from `from_block` through 12 (1-indexed)."""
    if not (1 <= int(from_block) <= VIT_B16_NUM_BLOCKS):
        raise ValueError(f"from_block must be in 1..{VIT_B16_NUM_BLOCKS}; got {from_block}")
    freeze_vit_encoder(encoder)
    layers = encoder.backbone.encoder.layers
    for layer in layers[int(from_block) - 1 :]:
        set_requires_grad(layer, True)
    set_requires_grad(encoder.backbone.encoder.ln, True)
    if include_fc:
        set_requires_grad(encoder.fc, True)
    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)


def freeze_resnet50_encoder(encoder: ResNet50Encoder) -> None:
    set_requires_grad(encoder, False)


def unfreeze_resnet50_encoder_tail(
    encoder: ResNet50Encoder,
    from_block: int = 4,
    include_fc: bool = True,
) -> int:
    """Unfreeze residual stages from `from_block` through layer4 (+ optional proj fc)."""
    if not (1 <= from_block <= RESNET50_NUM_STAGES):
        raise ValueError(
            f"from_block must be between 1 and {RESNET50_NUM_STAGES}; got {from_block}"
        )
    freeze_resnet50_encoder(encoder)
    for stage_idx in range(from_block, RESNET50_NUM_STAGES + 1):
        set_requires_grad(getattr(encoder, f"layer{stage_idx}"), True)
    if include_fc:
        set_requires_grad(encoder.avgpool, True)
        set_requires_grad(encoder.fc, True)
    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)


def lr_warmup_cosine(epoch: int, base_lr: float, total_epochs: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    if total_epochs <= warmup_epochs:
        return base_lr
    progress = float(epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_optimizer_param_group_lrs(optimizer: torch.optim.Optimizer, lrs: list[float]) -> None:
    for param_group, lr in zip(optimizer.param_groups, lrs):
        param_group["lr"] = lr


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_stage4_optimizer(
    model: FilmResidualFairnessModel,
    head_lr: float,
    encoder_lr: float,
    weight_decay: float,
    unfreeze_encoder: bool,
) -> torch.optim.Optimizer:
    head_modules = [model.demo_embedding, model.demo_gate, model.proposed_head]
    head_params = [p for m in head_modules for p in m.parameters() if p.requires_grad]
    encoder_params = [p for p in model.image_encoder.parameters() if p.requires_grad] if unfreeze_encoder else []
    if not head_params:
        raise RuntimeError("No trainable head parameters for Stage 4.")
    groups: list[dict[str, Any]] = [{"params": head_params, "lr": head_lr, "name": "head"}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr, "name": "encoder"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def multilabel_macro_auc_report(
    probs: torch.Tensor, labels: torch.Tensor, subgroups: torch.Tensor, num_subgroups: int
) -> dict[str, float]:
    """Macro AUC over the 5 competition pathologies, overall and per subgroup.

    ``probs``/``labels`` are expected in the 5-disease eval space (shape ``[N, 5]``).
    Keys match the FairVision binary-AUC report shape (``overall_auc``,
    ``subgroup_{g}_auc``) so downstream helpers (worst-subgroup, improvement gaps,
    etc.) work unchanged — the values are macro AUCs here, not single-label AUCs.
    """
    labels_np = labels.cpu().numpy()
    probs_np = probs.cpu().numpy()
    subgroups_np = subgroups.cpu().numpy()
    report: dict[str, float] = {}
    overall_aucs = _per_disease_aucs(labels_np, probs_np)
    report["overall_auc"] = float(np.mean(overall_aucs)) if overall_aucs else float("nan")
    for g in range(num_subgroups):
        mask = subgroups_np == g
        if mask.sum() > 1:
            g_aucs = _per_disease_aucs(labels_np[mask], probs_np[mask])
            report[f"subgroup_{g}_auc"] = float(np.mean(g_aucs)) if g_aucs else float("nan")
        else:
            report[f"subgroup_{g}_auc"] = float("nan")
    return report


binary_auc_report = multilabel_macro_auc_report


def worst_subgroup_auc(report: dict[str, float], num_subgroups: int) -> float:
    vals = [float(report.get(f"subgroup_{g}_auc", float("nan"))) for g in range(num_subgroups)]
    vals = [v for v in vals if not math.isnan(v)]
    return min(vals) if vals else float("nan")


def subgroup_auc_improvement_gap(
    base_report: dict[str, float], compare_report: dict[str, float], num_subgroups: int
) -> float:
    deltas = []
    for g in range(num_subgroups):
        b = float(base_report.get(f"subgroup_{g}_auc", float("nan")))
        c = float(compare_report.get(f"subgroup_{g}_auc", float("nan")))
        if not math.isnan(b) and not math.isnan(c):
            deltas.append(c - b)
    return float(max(deltas) - min(deltas)) if len(deltas) > 1 else float("nan")


def min_subgroup_improvement_delta(
    base_report: dict[str, float], compare_report: dict[str, float], num_subgroups: int
) -> float:
    """min_g Δ_g where Δ_g = AUC_compare(g) − AUC_baseline(g) on validation."""
    deltas = []
    for g in range(num_subgroups):
        b = float(base_report.get(f"subgroup_{g}_auc", float("nan")))
        c = float(compare_report.get(f"subgroup_{g}_auc", float("nan")))
        if not math.isnan(b) and not math.isnan(c):
            deltas.append(c - b)
    return float(min(deltas)) if deltas else float("nan")


def effective_selection_metric(stage_label: str, selection_metric: str) -> str:
    """min_improvement needs a frozen baseline (Stages 2/4); Stage 1 falls back to worst_subgroup."""
    if selection_metric == "min_improvement" and stage_label == "stage1":
        print(
            "[WARN] selection_metric=min_improvement is undefined for Stage 1 "
            "(no baseline reference); using worst_subgroup instead."
        )
        return "worst_subgroup"
    return selection_metric


def subgroup_improvement_summary(
    base_report: dict[str, float], compare_report: dict[str, float], num_subgroups: int
) -> dict[str, Any]:
    deltas = []
    for g in range(num_subgroups):
        b = float(base_report.get(f"subgroup_{g}_auc", float("nan")))
        c = float(compare_report.get(f"subgroup_{g}_auc", float("nan")))
        if not math.isnan(b) and not math.isnan(c):
            deltas.append(c - b)
    gap = float(max(deltas) - min(deltas)) if len(deltas) > 1 else float("nan")
    min_improvement = float(min(deltas)) if deltas else float("nan")
    return {"deltas": deltas, "gap": gap, "min_improvement": min_improvement}


def test_min_subgroup_improvement(test_eval: dict[str, Any]) -> float:
    """Worst-subgroup AUC gain vs Stage-1 baseline on test: min_g Δ_g."""
    summary = test_eval.get("proposed_vs_baseline", {})
    if "min_improvement" in summary and summary["min_improvement"] is not None:
        val = float(summary["min_improvement"])
        if not math.isnan(val):
            return val
    deltas = summary.get("deltas", [])
    if not deltas:
        return float("-inf")
    return float(min(deltas))


def select_stage4_checkpoint_by_test_min_improvement(
    test_results_by_metric: dict[str, Any],
) -> tuple[str, float]:
    """Pick among stage4_{overall_auc,worst_subgroup,min_improvement} by highest test min Δ_g."""
    best_metric: str | None = None
    best_min_imp = -float("inf")
    best_overall = -float("inf")
    for metric in STAGE4_CHECKPOINT_METRICS:
        if metric not in test_results_by_metric:
            continue
        tr = test_results_by_metric[metric]["test_results"]
        min_imp = test_min_subgroup_improvement(tr)
        overall = float(tr.get("proposed_test", {}).get("overall_auc", float("-inf")))
        if min_imp > best_min_imp or (min_imp == best_min_imp and overall > best_overall):
            best_min_imp = min_imp
            best_overall = overall
            best_metric = metric
    if best_metric is None:
        raise RuntimeError(
            "Could not select Stage-4 checkpoint: missing test results for "
            f"{STAGE4_CHECKPOINT_METRICS}"
        )
    return best_metric, best_min_imp


def subgroup_auc_gap(report: dict[str, float], num_subgroups: int) -> float:
    vals = [float(report.get(f"subgroup_{g}_auc", float("nan"))) for g in range(num_subgroups)]
    vals = [v for v in vals if not math.isnan(v)]
    return max(vals) - min(vals) if len(vals) > 1 else float("nan")


def _is_better(
    val_auc: float,
    val_primary: float,
    val_gap: float,
    best_auc: float,
    best_primary: float,
    best_gap: float,
    selection_metric: str,
    use_overall_tie_break: bool,
) -> bool:
    """
    Checkpoint selection.

    worst_subgroup: maximize worst-subgroup AUC; tie-break smaller improvement gap; then overall AUC.
    min_improvement: maximize min_g Δ_g vs baseline; tie-break smaller improvement gap; then overall AUC.
    overall_auc: maximize overall AUC only.
    """
    if selection_metric == "overall_auc":
        return (not math.isnan(val_auc)) and val_auc > best_auc
    if selection_metric in ("worst_subgroup", "min_improvement"):
        if math.isnan(val_primary):
            return False
        if val_primary > best_primary:
            return True
        if val_primary == best_primary:
            if not math.isnan(val_gap) and (math.isnan(best_gap) or val_gap < best_gap):
                return True
            gap_tied = val_gap == best_gap or (math.isnan(val_gap) and math.isnan(best_gap))
            if gap_tied and use_overall_tie_break:
                return (not math.isnan(val_auc)) and val_auc > best_auc
        return False
    raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}; got {selection_metric!r}")


@torch.no_grad()
def collect_predictions(
    model: FilmResidualFairnessModel,
    dataloader: DataLoader,
    device: torch.device,
    head: str = "baseline",
    usefulness_score: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sigmoid probs/labels in 5-disease eval space (for macro AUC)."""
    model.eval()
    all_probs, all_labels, all_groups = [], [], []
    for batch in dataloader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch.get("labels_eval", batch["labels"]).to(device, non_blocking=True)
        g = batch["subgroup"].to(device, non_blocking=True)
        out = model(x, g, usefulness_score=usefulness_score)
        logits = out[head]
        probs_e, y_e = to_eval_space(torch.sigmoid(logits), y)
        all_probs.append(probs_e.cpu())
        all_labels.append(y_e.cpu())
        all_groups.append(g.cpu())
    return torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_groups)


def sigmoid_np(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def compute_usefulness_score(
    baseline_report: dict[str, float],
    demo_report: dict[str, float],
    num_subgroups: int,
    alpha: float = 50.0,
    lambda_cv: float = 1.0,
    eps: float = 1e-8,
) -> tuple[float, dict[str, Any]]:
    """
    Utility / usefulness score u (paper §4.3).

    Eq. (4):  d = M(f) − M(f₀)                         overall gain
    Eq. (5):  Δ_g = M_g(f) − M_g(f₀)                    subgroup gain
    Eq. (6):  B = Σ_{Δ_g>0} Δ_g ,  H = Σ_{Δ_g<0} (−Δ_g) benefit / harm
              Δ_g⁺ = ReLU(Δ_g)
    Eq. (7):  CV = σ(Δ⁺) / (μ(Δ⁺) + ε) ,  P_cv = exp(−λ_cv · CV)
    Eq. (8):  u = 1[B > H] · σ(α d) · P_cv
    """
    # Eq. (4): overall gain d = M(f) − M(f₀)
    auc_b = float(baseline_report.get("overall_auc", float("nan")))
    auc_d = float(demo_report.get("overall_auc", float("nan")))
    d = auc_d - auc_b
    raw_deltas = []
    for g in range(num_subgroups):
        auc_bg = float(baseline_report.get(f"subgroup_{g}_auc", float("nan")))
        auc_dg = float(demo_report.get(f"subgroup_{g}_auc", float("nan")))
        if not math.isnan(auc_bg) and not math.isnan(auc_dg):
            # Eq. (5): Δ_g = M_g(f) − M_g(f₀)
            raw_deltas.append(auc_dg - auc_bg)

    improvement_gap = float(max(raw_deltas) - min(raw_deltas)) if len(raw_deltas) > 1 else float("nan")
    if len(raw_deltas) == 0 or math.isnan(d):
        return 0.0, {"d": d, "subgroup_auc_improvement_gap": improvement_gap, "gate_passed": 0.0}

    delta = np.asarray(raw_deltas, dtype=np.float64)
    # Eq. (6): B = Σ ReLU(Δ_g), H = Σ ReLU(−Δ_g);  Δ_g⁺ = ReLU(Δ_g)
    delta_pos = np.maximum(0.0, delta)
    delta_neg = np.maximum(0.0, -delta)
    B, H = float(delta_pos.sum()), float(delta_neg.sum())
    # Eq. (8) gate: 1[B > H]
    if B <= H:
        return 0.0, {"d": d, "B": B, "H": H, "subgroup_auc_improvement_gap": improvement_gap, "gate_passed": 0.0}

    mu = float(delta_pos.mean())
    sigma = float(delta_pos.std())
    if mu <= eps:
        return 0.0, {"d": d, "B": B, "H": H, "subgroup_auc_improvement_gap": improvement_gap, "gate_passed": 0.0}

    # Eq. (7): CV = σ(Δ⁺)/(μ(Δ⁺)+ε),  P_cv = exp(−λ_cv · CV)
    CV = sigma / (mu + eps)
    penalty = math.exp(-lambda_cv * CV)
    # Eq. (8): u = 1[B > H] · σ(α d) · P_cv
    usefulness = sigmoid_np(alpha * d) * penalty
    return float(usefulness), {
        "d": float(d),
        "B": B,
        "H": H,
        "CV": float(CV),
        "penalty": float(penalty),
        "subgroup_auc_improvement_gap": improvement_gap,
        "gate_passed": 1.0,
        "deltas_raw": delta.tolist(),
    }


def kd_loss_from_baseline(proposed_logits: torch.Tensor, baseline_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled soft-BCE KD (multi-label); alias kept for structural parity."""
    return kd_loss_multilabel(proposed_logits, baseline_logits, temperature)


def compute_subgroup_loss_improvements(
    proposed_logits: torch.Tensor,
    labels: torch.Tensor,
    subgroups: torch.Tensor,
    baseline_group_losses: torch.Tensor,
    num_subgroups: int,
    use_relu: bool = False,
) -> torch.Tensor | None:
    """
    Differentiable subgroup-improvement surrogate (paper §4.4).

    Eq. (14):  Δ^L_g = ℓ⁰_g − ℓ^{cf}_g
      (loss reduction vs frozen Stage-1 baseline; AUC is non-differentiable)

    Optional ReLU keeps only positive improvements (aligned with Δ_g⁺ in §4.3).
    Returns None when no subgroup is present in the batch.
    """
    improvements: list[torch.Tensor] = []
    for group_id in range(num_subgroups):
        mask = subgroups == group_id
        if mask.sum() == 0:
            continue
        # ℓ^{cf}_g
        proposed_loss_g = bce_logits_loss(proposed_logits[mask], labels[mask], reduction="mean")
        # Eq. (14): Δ^L_g = ℓ⁰_g − ℓ^{cf}_g
        improvement_g = baseline_group_losses[group_id].to(
            device=proposed_logits.device, dtype=proposed_logits.dtype
        ) - proposed_loss_g
        if use_relu:
            improvement_g = F.relu(improvement_g)
        improvements.append(improvement_g)
    if not improvements:
        return None
    return torch.stack(improvements)


def improvement_variance_loss(improvements: torch.Tensor) -> torch.Tensor:
    """Eq. (15): L_imp = Var_g(Δ^L_g)."""
    if improvements.numel() <= 1:
        return torch.zeros((), device=improvements.device, dtype=improvements.dtype)
    return ((improvements - improvements.mean()) ** 2).mean()


def improvement_worst_loss(improvements: torch.Tensor) -> torch.Tensor:
    """Eq. (16): L_worst = −min_g(Δ^L_g)."""
    return -improvements.min()


def improvement_fairness_loss(
    proposed_logits: torch.Tensor,
    labels: torch.Tensor,
    subgroups: torch.Tensor,
    baseline_group_losses: torch.Tensor,
    num_subgroups: int,
    use_relu: bool = False,
) -> torch.Tensor:
    """Backward-compatible alias: Var_g(Δ̃_g) only."""
    improvements = compute_subgroup_loss_improvements(
        proposed_logits, labels, subgroups, baseline_group_losses, num_subgroups, use_relu
    )
    if improvements is None:
        return torch.zeros((), device=proposed_logits.device, dtype=proposed_logits.dtype)
    return improvement_variance_loss(improvements)


def proposed_improvement_losses(
    proposed_logits: torch.Tensor,
    labels: torch.Tensor,
    subgroups: torch.Tensor,
    baseline_group_losses: torch.Tensor,
    num_subgroups: int,
    use_relu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Var_g(Δ̃_g), -min_g(Δ̃_g)) for Stage 4."""
    improvements = compute_subgroup_loss_improvements(
        proposed_logits, labels, subgroups, baseline_group_losses, num_subgroups, use_relu
    )
    if improvements is None:
        zero = torch.zeros((), device=proposed_logits.device, dtype=proposed_logits.dtype)
        return zero, zero
    return improvement_variance_loss(improvements), improvement_worst_loss(improvements)


def train_stage1_baseline(model, train_loader, val_loader, device, cfg: Config):
    print("\n========== Stage 1: Train baseline image-only head ==========")
    set_requires_grad(model.image_encoder, True)
    set_requires_grad(model.baseline_head, True)
    set_requires_grad(model.demo_embedding, False)
    set_requires_grad(model.demo_head, False)
    set_requires_grad(model.demo_gate, False)
    set_requires_grad(model.proposed_head, False)

    optimizer = make_optimizer(model, cfg.stage1_lr, cfg.weight_decay)
    best_auc, best_worst, best_state, best_epoch = -float("inf"), -float("inf"), copy.deepcopy(model.state_dict()), -1

    for epoch in range(cfg.stage1_epochs):
        if cfg.use_cosine_schedule_stage1:
            set_optimizer_param_group_lrs(
                optimizer, [lr_warmup_cosine(epoch, cfg.stage1_lr, cfg.stage1_epochs, 0, cfg.min_lr_ratio)]
            )
        model.train()
        loss_sum, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Stage 1 [{epoch + 1}/{cfg.stage1_epochs}]", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            g = batch["subgroup"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = bce_logits_loss(model(x, g)["baseline"], y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)

        val_probs, val_labels, val_groups = collect_predictions(model, val_loader, device, head="baseline")
        val_report = binary_auc_report(val_probs, val_labels, val_groups, cfg.num_subgroups)
        val_auc = val_report["overall_auc"]
        val_worst = worst_subgroup_auc(val_report, cfg.num_subgroups)
        stage1_metric = effective_selection_metric("stage1", cfg.selection_metric_stage1)
        print(f"Stage 1 Epoch {epoch + 1:03d}: Loss={loss_sum / max(n,1):.4f}, ValAUC={val_auc:.4f}, ValWorst={val_worst:.4f}")
        if _is_better(
            val_auc, val_worst, float("nan"), best_auc, best_worst, float("nan"),
            stage1_metric, cfg.use_overall_tie_break,
        ):
            best_auc, best_worst, best_epoch = val_auc, val_worst, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"Best Stage 1: epoch={best_epoch}, val_auc={best_auc:.4f}, val_worst={best_worst:.4f}")
    return model


def train_stage2_demo(
    model, train_loader, val_loader, device, cfg: Config, stage1_ref: Stage1BaselineReference
):
    print("\n========== Stage 2: Train demo head on [z;e] (for usefulness) ==========")
    set_requires_grad(model.image_encoder, False)
    set_requires_grad(model.baseline_head, False)
    set_requires_grad(model.demo_embedding, True)
    set_requires_grad(model.demo_head, True)
    set_requires_grad(model.demo_gate, False)
    set_requires_grad(model.proposed_head, False)

    optimizer = make_optimizer(model, cfg.stage2_lr, cfg.weight_decay)
    best_auc, best_worst, best_min_imp, best_gap = -float("inf"), -float("inf"), -float("inf"), float("inf")
    best_state, best_epoch = copy.deepcopy(model.state_dict()), -1
    stage2_metric = effective_selection_metric("stage2", cfg.selection_metric_stage2)

    baseline_val_report = stage1_ref.auc_report(val_loader, device, cfg.num_subgroups)

    for epoch in range(cfg.stage2_epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Stage 2 [{epoch + 1}/{cfg.stage2_epochs}]", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            g = batch["subgroup"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = bce_logits_loss(model(x, g)["demo"], y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)

        val_probs, val_labels, val_groups = collect_predictions(model, val_loader, device, head="demo")
        val_report = binary_auc_report(val_probs, val_labels, val_groups, cfg.num_subgroups)
        val_auc = val_report["overall_auc"]
        val_worst = worst_subgroup_auc(val_report, cfg.num_subgroups)
        val_gap = subgroup_auc_improvement_gap(baseline_val_report, val_report, cfg.num_subgroups)
        val_min_imp = min_subgroup_improvement_delta(baseline_val_report, val_report, cfg.num_subgroups)
        val_primary = val_min_imp if stage2_metric == "min_improvement" else val_worst
        best_primary = best_min_imp if stage2_metric == "min_improvement" else best_worst
        print(
            f"Stage 2 Epoch {epoch + 1:03d}: Loss={loss_sum / max(n,1):.4f}, ValAUC={val_auc:.4f}, "
            f"ValWorst={val_worst:.4f}, ValMinImp={val_min_imp:.4f}, ValGap={val_gap:.4f}"
        )
        if _is_better(
            val_auc, val_primary, val_gap, best_auc, best_primary, best_gap,
            stage2_metric, cfg.use_overall_tie_break,
        ):
            best_auc, best_gap, best_epoch = val_auc, val_gap, epoch + 1
            if stage2_metric == "min_improvement":
                best_min_imp = val_min_imp
            else:
                best_worst = val_worst
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if stage2_metric == "min_improvement":
        print(f"Best Stage 2: epoch={best_epoch}, val_auc={best_auc:.4f}, val_min_imp={best_min_imp:.4f}, val_gap={best_gap:.4f}")
    else:
        print(f"Best Stage 2: epoch={best_epoch}, val_auc={best_auc:.4f}, val_worst={best_worst:.4f}, val_gap={best_gap:.4f}")
    return model


def stage3_compute_usefulness(
    model, val_loader, device, cfg: Config, stage1_ref: Stage1BaselineReference
):
    print("\n========== Stage 3: Compute usefulness score (demo vs Stage-1 baseline) ==========")
    baseline_report = stage1_ref.auc_report(val_loader, device, cfg.num_subgroups)
    demo_probs, labels, groups = collect_predictions(model, val_loader, device, head="demo")
    demo_report = binary_auc_report(demo_probs, labels, groups, cfg.num_subgroups)
    usefulness_score, usefulness_meta = compute_usefulness_score(
        baseline_report, demo_report, cfg.num_subgroups, cfg.alpha, cfg.lambda_cv, cfg.eps
    )
    print("Baseline val:", baseline_report)
    print("Demo val:", demo_report)
    print(f"Usefulness score: {usefulness_score:.6f}")
    print("Usefulness meta:", usefulness_meta)
    return usefulness_score, usefulness_meta, baseline_report, demo_report


def _new_stage4_tracker() -> dict[str, Any]:
    return {
        "best_auc": -float("inf"),
        "best_primary": -float("inf"),
        "best_gap": float("inf"),
        "best_epoch": -1,
        "best_state": None,
        "val_metrics": {},
    }


def _stage4_val_primary(
    metric: str, val_auc: float, val_worst: float, val_min_imp: float
) -> float:
    if metric == "overall_auc":
        return val_auc
    if metric == "min_improvement":
        return val_min_imp
    return val_worst


def _update_stage4_tracker(
    tracker: dict[str, Any],
    metric: str,
    val_auc: float,
    val_worst: float,
    val_min_imp: float,
    val_gap: float,
    model_state: dict[str, torch.Tensor],
    epoch: int,
    use_overall_tie_break: bool,
) -> None:
    val_primary = _stage4_val_primary(metric, val_auc, val_worst, val_min_imp)
    if _is_better(
        val_auc,
        val_primary,
        val_gap,
        tracker["best_auc"],
        tracker["best_primary"],
        tracker["best_gap"],
        metric,
        use_overall_tie_break,
    ):
        tracker["best_auc"] = val_auc
        tracker["best_primary"] = val_primary
        tracker["best_gap"] = val_gap
        tracker["best_epoch"] = epoch
        tracker["best_state"] = copy.deepcopy(model_state)
        tracker["val_metrics"] = {
            "overall_auc": val_auc,
            "worst_subgroup_auc": val_worst,
            "min_subgroup_improvement": val_min_imp,
            "subgroup_auc_improvement_gap": val_gap,
        }


def build_stage4_checkpoint_bundle(trackers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    for metric in STAGE4_CHECKPOINT_METRICS:
        tracker = trackers[metric]
        if tracker["best_state"] is None:
            raise RuntimeError(f"Stage 4 did not produce a checkpoint for metric={metric}.")
        bundle[metric] = {
            "state": tracker["best_state"],
            "epoch": tracker["best_epoch"],
            "val_metrics": tracker["val_metrics"],
        }
    return bundle


def save_stage4_checkpoints(
    cfg: Config,
    bundle: dict[str, dict[str, Any]],
    base_dir: Path | None = None,
) -> dict[str, str]:
    """Save one checkpoint per Stage-4 selection metric; returns metric → path."""
    paths: dict[str, str] = {}
    for metric in STAGE4_CHECKPOINT_METRICS:
        info = bundle[metric]
        payload = {
            "model_state": {k: v.detach().cpu().clone() for k, v in info["state"].items()},
            "stage": f"stage4_{metric}",
            "selection_metric": metric,
            "checkpoint_epoch": info["epoch"],
            "val_metrics": info["val_metrics"],
            "seed": cfg.seed,
            "run_name": cfg.run_name,
            "demographic_mode": cfg.demographic_mode,
            "fusion_type": cfg.fusion_type,
            "architecture": "model_finalizing",
        }
        if base_dir is not None:
            out_dir = base_dir / metric
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / "checkpoint.pt"
        else:
            out_dir = scratch_checkpoint_dir(cfg, f"stage4_{metric}")
            ckpt_path = out_dir / "checkpoint.pt"
        torch.save(payload, ckpt_path)
        paths[metric] = str(ckpt_path.resolve())
        print(f"Saved Stage 4 checkpoint ({metric}) → {ckpt_path.resolve()}")
    return paths


def evaluate_all_stage4_checkpoints(
    model: FilmResidualFairnessModel,
    bundle: dict[str, dict[str, Any]],
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    usefulness_score: float,
    stage1_ref: Stage1BaselineReference,
) -> dict[str, Any]:
    """Evaluate the Stage-4 val-selected checkpoint(s) on test (for logging).

    Primary/final model uses validation ``cfg.selection_metric_stage4``
    (default / only tracked metric: min_improvement vs the Stage-1 baseline).
    """
    results: dict[str, Any] = {}
    for metric in STAGE4_CHECKPOINT_METRICS:
        info = bundle[metric]
        print(f"\n========== Test evaluation (Stage 4 checkpoint: {metric}) ==========")
        model.load_state_dict(info["state"])
        test_eval = final_evaluation(
            model, test_loader, device, cfg, usefulness_score, stage1_ref
        )
        min_imp = test_min_subgroup_improvement(test_eval)
        print(f"  Test min_subgroup_improvement (vs Stage-1 baseline): {min_imp:.4f}")
        results[metric] = {
            "checkpoint_epoch": info["epoch"],
            "val_metrics": info["val_metrics"],
            "test_results": test_eval,
            "test_min_subgroup_improvement": min_imp,
        }

    selected_metric = effective_selection_metric("stage4", cfg.selection_metric_stage4)
    if selected_metric not in results:
        selected_metric, _ = select_stage4_checkpoint_by_test_min_improvement(results)
    selected_min_imp = results[selected_metric]["test_min_subgroup_improvement"]
    results["_selection"] = {
        "criterion": f"val_{selected_metric}",
        "selected_checkpoint_metric": selected_metric,
        "test_min_subgroup_improvement": selected_min_imp,
        "val_selection_metrics": list(STAGE4_CHECKPOINT_METRICS),
    }
    model.load_state_dict(bundle[selected_metric]["state"])
    print(
        f"\n========== Final Stage-4 model (val-selected: {selected_metric}) =========="
        f"\nCheckpoint: stage4_{selected_metric} | test min_subgroup_improvement={selected_min_imp:.4f}"
    )
    return results


def export_all_stage4_predictions(
    model: FilmResidualFairnessModel,
    bundle: dict[str, dict[str, Any]],
    cfg: Config,
    device: torch.device,
    run_dir: Path,
    usefulness_score: float,
    selected_metric: str | None = None,
    test_loader: DataLoader | None = None,
) -> dict[str, str]:
    """Write a single ``test_predictions.csv`` for the val-selected Stage-4 checkpoint."""
    if test_loader is None:
        _, _, test_loader = build_dataloaders(cfg)
    primary = selected_metric or effective_selection_metric("stage4", cfg.selection_metric_stage4)
    if primary not in bundle:
        raise KeyError(f"Selected Stage-4 metric {primary!r} missing from checkpoint bundle")
    model.load_state_dict(bundle[primary]["state"])
    csv_path = run_dir / "test_predictions.csv"
    export_test_predictions_csv(
        model, cfg, device, csv_path, usefulness_score, test_loader=test_loader, seed=cfg.seed
    )
    return {
        "test_predictions": str(csv_path.resolve()),
        "primary": str(csv_path.resolve()),
        "selected_checkpoint_metric": primary,
    }


def train_stage4_proposed(
    model,
    train_loader,
    val_loader,
    device,
    cfg,
    usefulness_score,
    stage1_ref: Stage1BaselineReference,
    save_checkpoints: bool = True,
):
    print(f"\n========== Stage 4: Train {model.fusion_type} fusion + proposed head ==========")
    print(
        f"Stage 4 loss (Stage-1 baseline reference): "
        f"L_task + {cfg.lambda_imp}·Var(Δ̃_g) + {cfg.lambda_worst}·(-min Δ̃_g) + {cfg.lambda_kd}·L_kd"
    )
    baseline_group_losses = stage1_ref.compute_subgroup_losses(train_loader, device, cfg.num_subgroups)
    print("Stage-1 baseline subgroup losses (train):", baseline_group_losses.cpu().tolist())
    set_requires_grad(model.image_encoder, False)
    set_requires_grad(model.baseline_head, False)
    set_requires_grad(model.demo_head, False)
    set_requires_grad(model.demo_embedding, True)
    set_requires_grad(model.demo_gate, True)
    set_requires_grad(model.proposed_head, True)

    if cfg.stage4_unfreeze_encoder:
        n_enc = unfreeze_encoder_tail(model.image_encoder, cfg)
        print(
            f"Stage 4 encoder unfreeze ({normalize_backbone(cfg.backbone)}): "
            f"block>={cfg.stage4_unfreeze_from_block}, params={n_enc:,}"
        )
    else:
        print("Stage 4 encoder unfreeze: disabled")

    optimizer = make_stage4_optimizer(
        model, cfg.stage4_head_lr, cfg.stage4_encoder_lr, cfg.weight_decay, cfg.stage4_unfreeze_encoder
    )
    trackers = {metric: _new_stage4_tracker() for metric in STAGE4_CHECKPOINT_METRICS}
    baseline_val_report = stage1_ref.auc_report(val_loader, device, cfg.num_subgroups)
    if cfg.stage4_epochs <= cfg.stage4_checkpoint_skip_epochs:
        raise ValueError(
            f"stage4_epochs ({cfg.stage4_epochs}) must be > stage4_checkpoint_skip_epochs "
            f"({cfg.stage4_checkpoint_skip_epochs}) to select checkpoints"
        )
    print(
        f"Stage 4 checkpoint selection: epochs 1–{cfg.stage4_checkpoint_skip_epochs} skipped; "
        f"eligible from epoch {cfg.stage4_checkpoint_skip_epochs + 1}"
    )

    for epoch in range(cfg.stage4_epochs):
        if cfg.use_cosine_schedule_stage4:
            head_lr = lr_warmup_cosine(epoch, cfg.stage4_head_lr, cfg.stage4_epochs, cfg.warmup_epochs_stage4, cfg.min_lr_ratio)
            if cfg.stage4_unfreeze_encoder and len(optimizer.param_groups) > 1:
                enc_lr = lr_warmup_cosine(epoch, cfg.stage4_encoder_lr, cfg.stage4_epochs, cfg.warmup_epochs_stage4, cfg.min_lr_ratio)
                set_optimizer_param_group_lrs(optimizer, [head_lr, enc_lr])
            else:
                set_optimizer_param_group_lrs(optimizer, [head_lr])

        model.train()
        loss_sum = task_sum = imp_sum = worst_sum = kd_sum = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"Stage 4 [{epoch + 1}/{cfg.stage4_epochs}]", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)  # train width (5 or 13)
            g = batch["subgroup"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x, g, usefulness_score=usefulness_score)
            logits_baseline = stage1_ref(x).detach()
            logits_proposed = out["proposed"]
            # Eq. (13): L_task = CE(ŷ_cf, y)  (BCE-with-logits for multi-label CheXpert)
            loss_task = bce_logits_loss(logits_proposed, y)
            # Eq. (15): L_imp = Var_g(Δ^L_g);  Eq. (16): L_worst = −min_g(Δ^L_g)  (Δ^L_g from Eq. 14)
            loss_imp, loss_worst = proposed_improvement_losses(
                logits_proposed, y, g, baseline_group_losses, cfg.num_subgroups, cfg.fairness_use_relu
            )
            # Eq. (17): L_kd
            loss_kd = kd_loss_from_baseline(logits_proposed, logits_baseline, cfg.temperature)
            # Eq. (18): L_total = L_task + λ_imp L_imp + λ_worst L_worst + λ_kd L_kd
            loss = (
                loss_task
                + cfg.lambda_imp * loss_imp
                + cfg.lambda_worst * loss_worst
                + cfg.lambda_kd * loss_kd
            )
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            loss_sum += loss.item() * bs
            task_sum += loss_task.item() * bs
            imp_sum += loss_imp.item() * bs
            worst_sum += loss_worst.item() * bs
            kd_sum += loss_kd.item() * bs
            n += bs

        val_probs, val_labels, val_groups = collect_predictions(
            model, val_loader, device, head="proposed", usefulness_score=usefulness_score
        )
        val_report = binary_auc_report(val_probs, val_labels, val_groups, cfg.num_subgroups)
        val_auc = val_report["overall_auc"]
        val_worst = worst_subgroup_auc(val_report, cfg.num_subgroups)
        val_gap = subgroup_auc_improvement_gap(baseline_val_report, val_report, cfg.num_subgroups)
        val_min_imp = min_subgroup_improvement_delta(baseline_val_report, val_report, cfg.num_subgroups)
        print(
            f"Stage 4 Epoch {epoch + 1:03d}: Loss={loss_sum / max(n,1):.4f}, Task={task_sum / max(n,1):.4f}, "
            f"ImpVar={imp_sum / max(n,1):.4f}, ImpWorst={worst_sum / max(n,1):.4f}, KD={kd_sum / max(n,1):.4f}, "
            f"ValAUC={val_auc:.4f}, ValWorst={val_worst:.4f}, ValMinImp={val_min_imp:.4f}, ValGap={val_gap:.4f}"
        )
        if epoch >= cfg.stage4_checkpoint_skip_epochs:
            current_state = copy.deepcopy(model.state_dict())
            for metric in STAGE4_CHECKPOINT_METRICS:
                _update_stage4_tracker(
                    trackers[metric],
                    metric,
                    val_auc,
                    val_worst,
                    val_min_imp,
                    val_gap,
                    current_state,
                    epoch + 1,
                    cfg.use_overall_tie_break,
                )

    bundle = build_stage4_checkpoint_bundle(trackers)
    checkpoint_paths: dict[str, str] = {}
    if save_checkpoints:
        checkpoint_paths = save_stage4_checkpoints(cfg, bundle)
    for metric in STAGE4_CHECKPOINT_METRICS:
        info = bundle[metric]
        vm = info["val_metrics"]
        path_note = f" → {checkpoint_paths[metric]}" if metric in checkpoint_paths else ""
        print(
            f"Best Stage 4 ({metric}): epoch={info['epoch']}, "
            f"val_auc={vm['overall_auc']:.4f}, val_worst={vm['worst_subgroup_auc']:.4f}, "
            f"val_min_imp={vm['min_subgroup_improvement']:.4f}, val_gap={vm['subgroup_auc_improvement_gap']:.4f}"
            f"{path_note}"
        )
    return model, bundle, checkpoint_paths


def final_evaluation(
    model, test_loader, device, cfg, usefulness_score, stage1_ref: Stage1BaselineReference
):
    print("\n========== Final Test Evaluation ==========")
    base_stage1_probs, labels, groups = stage1_ref.collect_predictions(test_loader, device)
    base_probs, _, _ = collect_predictions(model, test_loader, device, head="baseline")
    demo_probs, _, _ = collect_predictions(model, test_loader, device, head="demo")
    prop_probs, _, _ = collect_predictions(model, test_loader, device, head="proposed", usefulness_score=usefulness_score)
    baseline_test = binary_auc_report(base_stage1_probs, labels, groups, cfg.num_subgroups)
    baseline_post_stage4_test = binary_auc_report(base_probs, labels, groups, cfg.num_subgroups)
    demo_test = binary_auc_report(demo_probs, labels, groups, cfg.num_subgroups)
    proposed_test = binary_auc_report(prop_probs, labels, groups, cfg.num_subgroups)
    results = {
        "baseline_test": baseline_test,
        "baseline_post_stage4_test": baseline_post_stage4_test,
        "demo_test": demo_test,
        "proposed_test": proposed_test,
        "demo_vs_baseline": subgroup_improvement_summary(baseline_test, demo_test, cfg.num_subgroups),
        "proposed_vs_baseline": subgroup_improvement_summary(baseline_test, proposed_test, cfg.num_subgroups),
        "demo_vs_post_stage4_baseline": subgroup_improvement_summary(
            baseline_post_stage4_test, demo_test, cfg.num_subgroups
        ),
        "proposed_vs_post_stage4_baseline": subgroup_improvement_summary(
            baseline_post_stage4_test, proposed_test, cfg.num_subgroups
        ),
    }
    print("Baseline test (Stage 1):", baseline_test)
    print("Baseline test (post-Stage 4 encoder):", baseline_post_stage4_test)
    print("Demo test:", demo_test)
    print("Proposed test:", proposed_test)
    return results


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def seed_tag(cfg: Config) -> str:
    return f"seed_{cfg.seed}"


def results_run_dir(cfg: Config) -> Path:
    return Path(cfg.output_dir) / "results" / cfg.run_name / cfg.demographic_mode / seed_tag(cfg)


def scratch_checkpoint_dir(cfg: Config, stage: str) -> Path:
    path = Path(cfg.checkpoint_dir) / cfg.run_name / cfg.demographic_mode / seed_tag(cfg) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_stage_checkpoint(cfg: Config, model: FilmResidualFairnessModel, stage: str) -> Path:
    out_dir = scratch_checkpoint_dir(cfg, stage)
    payload = {
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "stage": stage,
        "seed": cfg.seed,
        "run_name": cfg.run_name,
        "demographic_mode": cfg.demographic_mode,
        "fusion_type": cfg.fusion_type,
        "architecture": "model_finalizing",
    }
    torch.save(payload, out_dir / "checkpoint.pt")
    print(f"Saved {stage} checkpoint → {(out_dir / 'checkpoint.pt').resolve()}")
    return out_dir


def save_stage1_baseline_reference(cfg: Config, stage1_ref: Stage1BaselineReference) -> Path:
    out_dir = scratch_checkpoint_dir(cfg, "stage1_baseline_ref")
    payload = {
        "encoder_state": {k: v.detach().cpu().clone() for k, v in stage1_ref.image_encoder.state_dict().items()},
        "baseline_head_state": {
            k: v.detach().cpu().clone() for k, v in stage1_ref.baseline_head.state_dict().items()
        },
        "stage": "stage1_baseline_ref",
        "seed": cfg.seed,
        "run_name": cfg.run_name,
        "demographic_mode": cfg.demographic_mode,
        "architecture": "model_finalizing",
    }
    torch.save(payload, out_dir / "checkpoint.pt")
    print(f"Saved Stage-1 baseline reference → {(out_dir / 'checkpoint.pt').resolve()}")
    return out_dir


def build_stage1_baseline_reference(
    cfg: Config,
    model: ProposedFairnessModel,
    device: torch.device,
    save: bool = True,
) -> Stage1BaselineReference:
    ref = Stage1BaselineReference.from_model(model).to(device)
    if save:
        save_stage1_baseline_reference(cfg, ref)
    return ref


def save_run_results(cfg: Config, all_results: dict[str, Any]) -> Path:
    out_dir = results_run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(asdict(cfg)), f, indent=2)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(all_results), f, indent=2)
    torch.save(all_results, out_dir / "results.pt")
    print(f"Saved results → {out_dir.resolve()}")
    return out_dir


def resolve_primary_checkpoint(cfg: Config) -> Path:
    for stage in ("stage4", "stage5"):
        path = scratch_checkpoint_dir(cfg, stage) / "checkpoint.pt"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing stage4 checkpoint under {scratch_checkpoint_dir(cfg, 'stage4')}")


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def normalize_backbone(backbone: str) -> str:
    name = str(backbone).strip().lower()
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"backbone must be one of {SUPPORTED_BACKBONES}; got {backbone!r}")
    return name


def build_image_encoder(cfg: Config) -> nn.Module:
    backbone = normalize_backbone(cfg.backbone)
    if backbone == "resnet50":
        return ResNet50Encoder(cfg.feature_dim, cfg.image_channels, cfg.pretrained)
    if backbone == "vit_b_16":
        return ViTB16Encoder(cfg.feature_dim, cfg.image_channels, cfg.pretrained)
    raise ValueError(f"Unsupported backbone {backbone!r}")


def unfreeze_encoder_tail(encoder: nn.Module, cfg: Config) -> int:
    backbone = normalize_backbone(cfg.backbone)
    if backbone == "resnet50":
        return unfreeze_resnet50_encoder_tail(encoder, cfg.stage4_unfreeze_from_block, cfg.stage4_unfreeze_include_fc)
    if backbone == "vit_b_16":
        return unfreeze_vit_encoder_tail(encoder, cfg.stage4_unfreeze_from_block, cfg.stage4_unfreeze_include_fc)
    raise ValueError(f"Unsupported backbone {backbone!r}")


def build_model(cfg: Config, device: torch.device) -> ProposedFairnessModel:
    encoder = build_image_encoder(cfg)
    return ProposedFairnessModel(
        image_encoder=encoder,
        feature_dim=cfg.feature_dim,
        num_subgroups=cfg.num_subgroups,
        demo_embed_dim=cfg.demo_embed_dim,
        num_classes=cfg.num_classes,
        gamma=cfg.gamma,
        fusion_type=cfg.fusion_type,
        residual_blocks=cfg.residual_blocks,
        head_layernorm=cfg.head_layernorm,
        head_hidden_dim=cfg.head_hidden_dim,
        head_dropout=cfg.head_dropout,
    ).to(device)


def model_artifact_name(cfg: Config) -> str:
    return f"{normalize_backbone(cfg.backbone)}_proposed_{cfg.demographic_mode}_fusion_{cfg.fusion_type}"


PHASE0_PREDICTION_CSV_FIELDS: tuple[str, ...] = (
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
)


class Stage1ImageOnlyExport(nn.Module):
    """Standalone image-only model (encoder + baseline head) for Phase-0 export.

    Reuses the already-trained Stage-1 modules from ``ProposedFairnessModel``
    (``image_encoder`` + ``baseline_head``) so the checkpoint can be reloaded
    without the full ContextFair model -- needed when ``--skip-baseline``
    reuses an existing seed's baseline without retraining.
    """

    def __init__(self, image_encoder: nn.Module, baseline_head: nn.Module):
        super().__init__()
        self.image_encoder = image_encoder
        self.baseline_head = baseline_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.baseline_head(self.image_encoder(x))


def phase0_erm_ckpt_path(output_dir: str | Path, backbone: str, seed: int) -> Path:
    return Path(output_dir) / "checkpoints" / "ERM" / f"seed_{seed}" / f"{backbone}_image_only.pt"


def phase0_erm_val_metrics_path(output_dir: str | Path, backbone: str, seed: int) -> Path:
    return Path(output_dir) / "checkpoints" / "ERM" / f"seed_{seed}" / f"{backbone}_image_only_val_metrics.json"


def phase0_erm_csv_path(output_dir: str | Path, backbone: str, seed: int) -> Path:
    return Path(output_dir) / "results" / "ERM" / f"seed_{seed}" / "test_predictions.csv"


@torch.no_grad()
def collect_phase0_test_rows(
    model: nn.Module, loader: DataLoader, device: torch.device, model_name: str, seed: int
) -> list[dict[str, Any]]:
    """One row per (sample x eval pathology), matching CheXpert ERM/Fusion/fairness CSVs."""
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch.get("labels_eval", batch["labels"]).to(device, non_blocking=True)
        logits = model(x)
        logits_e, y_e = to_eval_space(logits, y)
        probs_e = torch.sigmoid(logits_e)
        pred_e = (probs_e >= 0.5).long()

        sample_ids = batch["sample_id"]
        rel_paths = batch["rel_path"]
        sexes = batch["sex"].tolist()
        ages = batch["age"].tolist()
        labels_cpu = y_e.cpu()
        logits_cpu = logits_e.cpu()
        probs_cpu = probs_e.cpu()
        pred_cpu = pred_e.cpu()

        for i in range(len(sample_ids)):
            for j, pathology in enumerate(EVAL_DISEASE_NAMES):
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


def write_phase0_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PHASE0_PREDICTION_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export_phase0_erm_baseline(
    cfg: Config,
    model: ProposedFairnessModel,
    device: torch.device,
    val_loader: DataLoader,
    test_loader: DataLoader,
    seed: int | None = None,
) -> None:
    """Export the trained Stage-1 baseline as a fairness-style ERM artifact.

    Writes ``checkpoints/ERM/seed_{seed}/{backbone}_image_only.pt`` (+ val
    metrics JSON) and ``results/ERM/seed_{seed}/test_predictions.csv`` with
    ``model_name={backbone}_image_only``, matching
    ``../fairness/chexpert_fairness_common.py::ensure_baseline_erm``.
    """
    seed = int(seed if seed is not None else cfg.seed)
    backbone = normalize_backbone(cfg.backbone)
    ckpt_path = phase0_erm_ckpt_path(cfg.output_dir, backbone, seed)
    csv_path = phase0_erm_csv_path(cfg.output_dir, backbone, seed)

    if ckpt_path.is_file() and csv_path.is_file() and cfg.skip_baseline:
        print(f"[Phase0] ERM baseline exists for seed={seed} (skip_baseline=1): {ckpt_path}")
        return

    export_model = Stage1ImageOnlyExport(model.image_encoder, model.baseline_head).to(device)
    export_model.eval()

    val_probs, val_labels, val_groups = collect_predictions(model, val_loader, device, head="baseline")
    val_report = binary_auc_report(val_probs, val_labels, val_groups, cfg.num_subgroups)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_model.state_dict(), ckpt_path)
    with open(phase0_erm_val_metrics_path(cfg.output_dir, backbone, seed), "w", encoding="utf-8") as f:
        json.dump(
            to_jsonable(
                {
                    "seed": seed,
                    "backbone": backbone,
                    "demographic_mode": cfg.demographic_mode,
                    "hparams": {
                        "lr": cfg.stage1_lr,
                        "weight_decay": cfg.weight_decay,
                        "batch_size": cfg.batch_size,
                        "epochs": cfg.stage1_epochs,
                    },
                    "val_metrics": val_report,
                }
            ),
            f,
            indent=2,
        )

    rows = collect_phase0_test_rows(export_model, test_loader, device, f"{backbone}_image_only", seed)
    write_phase0_csv(csv_path, rows)
    print(f"[Phase0] Wrote ERM baseline ckpt={ckpt_path} csv={csv_path}")


def run_phase0_baselines(cfg: Config) -> None:
    """Fairness-style Phase 0: local image-only ERM baseline(s) for cfg.seeds.

    Falls back to ``[cfg.seed]`` when ``cfg.seeds`` is unset (ContextFair shells
    invoke this once per seed via ``--baseline-only --seed N``). For each seed:
    if the ERM checkpoint already exists and (``skip_baseline`` or the CSV is
    also present), skip retraining -- exporting the CSV from the existing
    checkpoint if it is missing. Otherwise train Stage 1 from scratch and
    export it. Missing checkpoint with ``skip_baseline=1`` raises, matching
    ``ensure_baseline_erm``.
    """
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    seeds = tuple(int(s) for s in (getattr(cfg, "seeds", ()) or (cfg.seed,)))
    backbone = normalize_backbone(cfg.backbone)

    erm_hparams = load_erm_stage1_hparams(cfg, backbone=backbone)
    if erm_hparams:
        apply_erm_hparams_to_cfg(cfg, erm_hparams)
    else:
        print(
            f"[WARN] Phase0: no ERM finetune summary found (erm_hparams_dir={cfg.erm_hparams_dir!r}); "
            f"using CLI Stage-1 hparams (lr={cfg.stage1_lr}, weight_decay={cfg.weight_decay}, "
            f"batch_size={cfg.batch_size}, epochs={cfg.stage1_epochs}). Production runs should point "
            "--erm-hparams-dir at ERM finetune summaries."
        )

    print(f"\n========== Phase 0: image-only ERM baseline ({backbone}) ==========")
    print(f"Seeds: {seeds} | demographic_mode={cfg.demographic_mode}")

    for seed in seeds:
        ckpt_path = phase0_erm_ckpt_path(cfg.output_dir, backbone, seed)
        csv_path = phase0_erm_csv_path(cfg.output_dir, backbone, seed)
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg.seed = seed

        if ckpt_path.is_file() and (cfg.skip_baseline or csv_path.is_file()):
            print(f"[Phase0] seed={seed}: ERM ckpt exists ({ckpt_path}); skip retrain.")
            if not csv_path.is_file():
                print(f"[Phase0] seed={seed}: CSV missing; exporting from existing ckpt.")
                _, _, test_loader = build_dataloaders(seed_cfg)
                probe = build_model(seed_cfg, device)
                export_model = Stage1ImageOnlyExport(
                    build_image_encoder(seed_cfg).to(device), copy.deepcopy(probe.baseline_head).to(device)
                )
                export_model.load_state_dict(torch.load(ckpt_path, map_location=device))
                export_model.eval()
                rows = collect_phase0_test_rows(export_model, test_loader, device, f"{backbone}_image_only", seed)
                write_phase0_csv(csv_path, rows)
                print(f"[Phase0] Wrote ERM baseline csv={csv_path} (from existing ckpt)")
            continue

        if cfg.skip_baseline and not ckpt_path.is_file():
            raise FileNotFoundError(
                f"skip_baseline=1 but missing image-only ERM checkpoint for seed={seed}: {ckpt_path}"
            )

        print(f"[Phase0] seed={seed}: training Stage-1 image-only baseline from scratch.")
        set_seed(seed)
        train_loader, val_loader, test_loader = build_dataloaders(seed_cfg)
        model = build_model(seed_cfg, device)
        model = train_stage1_baseline(model, train_loader, val_loader, device, seed_cfg)
        save_stage_checkpoint(seed_cfg, model, "stage1")
        build_stage1_baseline_reference(seed_cfg, model, device)
        export_phase0_erm_baseline(seed_cfg, model, device, val_loader, test_loader, seed=seed)

    print(f"Phase 0 complete for seeds {seeds}.")


PREDICTION_CSV_FIELDS: tuple[str, ...] = (
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
)


@torch.no_grad()
def export_test_predictions_csv(
    model,
    cfg: Config,
    device: torch.device,
    output_csv: Path,
    usefulness_score: float,
    test_loader: DataLoader | None = None,
    seed: int | None = None,
):
    """Write one row per (sample x eval pathology), matching CheXpert Fusion/fairness CSVs."""
    if test_loader is None:
        _, _, test_loader = build_dataloaders(cfg)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model_name = model_artifact_name(cfg)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_CSV_FIELDS)
        writer.writeheader()
        row_idx = 0
        for batch in test_loader:
            x = batch["image"].to(device, non_blocking=True)
            y_eval = batch.get("labels_eval", batch["labels"]).to(device, non_blocking=True)
            g = batch["subgroup"].to(device, non_blocking=True)
            sex_vals = batch.get("sex")
            age_vals = batch.get("age")
            sample_ids = batch.get("sample_id")
            rel_paths = batch.get("rel_path")
            logits = model(x, g, usefulness_score=usefulness_score)["proposed"]
            probs_e, y_e = to_eval_space(torch.sigmoid(logits), y_eval)
            logits_e, _ = to_eval_space(logits, None)
            bs = x.size(0)
            for i in range(bs):
                sample_id = sample_ids[i] if sample_ids is not None else str(row_idx)
                rel_path = rel_paths[i] if rel_paths is not None else ""
                sex_val = int(sex_vals[i].item()) if sex_vals is not None else ""
                age_val = int(age_vals[i].item()) if age_vals is not None else ""
                for j, pathology in enumerate(EVAL_DISEASE_NAMES):
                    prob = float(probs_e[i, j].item())
                    logit = float(logits_e[i, j].item())
                    true_label = int(y_e[i, j].item())
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "rel_path": rel_path,
                            "pathology": pathology,
                            "pathology_idx": j,
                            "true_label": true_label,
                            "logit": logit,
                            "prediction_probability": prob,
                            "pred_class": int(prob >= 0.5),
                            "sex": sex_val,
                            "age": age_val,
                            "model_name": model_name,
                            "seed": seed if seed is not None else "",
                        }
                    )
                row_idx += 1
    return output_csv


def finetune_results_dir(cfg: Config) -> Path:
    return Path(cfg.output_dir) / "results" / cfg.run_name / cfg.demographic_mode / "finetune"


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_matching_state_dict(model: nn.Module, source_state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    current_state = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in source_state.items():
        if key in current_state and current_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)
    current_state.update(matched)
    model.load_state_dict(current_state)
    return {"loaded": sorted(matched.keys()), "skipped": sorted(skipped)}


@torch.no_grad()
def evaluate_head_report(
    model: FilmResidualFairnessModel,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    head: str,
    usefulness_score: float = 0.0,
) -> dict[str, float]:
    probs, labels, groups = collect_predictions(model, val_loader, device, head=head, usefulness_score=usefulness_score)
    return binary_auc_report(probs, labels, groups, cfg.num_subgroups)


def report_selection_metrics(
    report: dict[str, float],
    cfg: Config,
    improvement_base_report: dict[str, float] | None = None,
) -> dict[str, float]:
    metrics = {
        "overall_auc": float(report.get("overall_auc", float("nan"))),
        "worst_subgroup_auc": worst_subgroup_auc(report, cfg.num_subgroups),
    }
    if improvement_base_report is not None:
        metrics["subgroup_auc_improvement_gap"] = subgroup_auc_improvement_gap(
            improvement_base_report, report, cfg.num_subgroups
        )
        metrics["min_subgroup_improvement"] = min_subgroup_improvement_delta(
            improvement_base_report, report, cfg.num_subgroups
        )
    else:
        metrics["subgroup_auc_improvement_gap"] = float("nan")
        metrics["min_subgroup_improvement"] = float("nan")
    return metrics


def _selection_primary(metrics: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "overall_auc":
        return float(metrics.get("overall_auc", float("nan")))
    if selection_metric == "min_improvement":
        return float(metrics.get("min_subgroup_improvement", float("nan")))
    return float(metrics.get("worst_subgroup_auc", float("nan")))


def is_better_metrics(
    metrics: dict[str, float],
    best_metrics: dict[str, float] | None,
    selection_metric: str,
    use_overall_tie_break: bool,
) -> bool:
    if best_metrics is None:
        return True
    return _is_better(
        metrics["overall_auc"],
        _selection_primary(metrics, selection_metric),
        metrics.get("subgroup_auc_improvement_gap", float("nan")),
        best_metrics["overall_auc"],
        _selection_primary(best_metrics, selection_metric),
        best_metrics.get("subgroup_auc_improvement_gap", float("nan")),
        selection_metric,
        use_overall_tie_break,
    )


def format_float_for_path(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def print_finetune_trial(stage: str, hparams: dict[str, Any]) -> None:
    hparam_str = ", ".join(f"{k}={v}" for k, v in hparams.items())
    print(f"\n[{stage}] Finetune trial: {hparam_str}")


def read_finetune_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Finetune summary not found: {summary_path}")
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def apply_summary_hparams_to_cfg(cfg: Config, summary: dict[str, Any]) -> None:
    """Apply saved stage hparams from finetune_summary.json onto cfg."""
    if "stage1" in summary:
        s1 = summary["stage1"]["hparams"]
        cfg.stage1_lr = float(s1["lr"])
        cfg.weight_decay = float(s1["weight_decay"])
        if "batch_size" in s1:
            cfg.batch_size = int(s1["batch_size"])
    if "stage2" in summary:
        cfg.demo_embed_dim = int(summary["stage2"]["hparams"]["demo_embed_dim"])
    s3 = summary.get("stage3", {}).get("hparams", {})
    if "alpha" in s3:
        cfg.alpha = float(s3["alpha"])
    if "lambda_cv" in s3:
        cfg.lambda_cv = float(s3["lambda_cv"])
    s4 = summary.get("stage4", summary.get("stage5", {})).get("hparams", {})
    for key, attr in (
        ("lambda_imp", "lambda_imp"),
        ("lambda_worst", "lambda_worst"),
        ("lambda_kd", "lambda_kd"),
        ("temperature", "temperature"),
        ("gamma", "gamma"),
        ("stage4_head_lr", "stage4_head_lr"),
        ("stage4_encoder_lr", "stage4_encoder_lr"),
        ("stage5_head_lr", "stage4_head_lr"),
        ("stage5_encoder_lr", "stage4_encoder_lr"),
    ):
        if key in s4:
            setattr(cfg, attr, float(s4[key]))
    if "stage4_epochs" in s4:
        cfg.stage4_epochs = int(s4["stage4_epochs"])
    elif "stage5_epochs" in s4:
        cfg.stage4_epochs = int(s4["stage5_epochs"])
    if "stage4_unfreeze_from_block" in s4:
        cfg.stage4_unfreeze_from_block = int(s4["stage4_unfreeze_from_block"])
    elif "stage5_unfreeze_from_block" in s4:
        cfg.stage4_unfreeze_from_block = int(s4["stage5_unfreeze_from_block"])


def load_stage2_state_from_checkpoint(cfg: Config) -> dict[str, torch.Tensor]:
    path = scratch_checkpoint_dir(cfg, "stage2") / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage-2 checkpoint not found: {path}. "
            "Run Stages 1–2 finetune first with checkpoint saving enabled."
        )
    return load_checkpoint_state(path)


def _load_stages_through_3_from_artifacts(
    cfg: Config,
    val_loader: DataLoader,
    device: torch.device,
    summary: dict[str, Any],
) -> tuple[
    Stage1BaselineReference,
    dict[str, torch.Tensor],
    Config,
    dict[str, float],
    float,
    dict[str, Any],
    dict[str, Any],
]:
    """Restore Stages 1–3 from summary + on-disk checkpoints (no retraining)."""
    apply_summary_hparams_to_cfg(cfg, summary)
    stage1_ref = Stage1BaselineReference.from_checkpoint(cfg, device)
    stage2_state = load_stage2_state_from_checkpoint(cfg)
    stage1_baseline_val_report = stage1_ref.auc_report(val_loader, device, cfg.num_subgroups)

    model = build_model(cfg, device)
    load_matching_state_dict(model, stage2_state)
    demo_report = evaluate_head_report(model, val_loader, device, cfg, head="demo")
    s3h = summary.get("stage3", {}).get("hparams", {})
    alpha = float(s3h.get("alpha", cfg.alpha))
    lambda_cv = float(s3h.get("lambda_cv", cfg.lambda_cv))
    usefulness_score, usefulness_meta = compute_usefulness_score(
        stage1_baseline_val_report,
        demo_report,
        cfg.num_subgroups,
        alpha,
        lambda_cv,
        cfg.eps,
    )
    early_summary = {
        "stage1": summary.get("stage1", {}),
        "stage2": summary.get("stage2", {}),
        "stage3": {
            "hparams": {"alpha": alpha, "lambda_cv": lambda_cv},
            "metrics": {
                "usefulness_score": usefulness_score,
                **usefulness_meta,
            },
            "split": "val",
        },
    }
    return (
        stage1_ref,
        stage2_state,
        cfg,
        stage1_baseline_val_report,
        usefulness_score,
        usefulness_meta,
        early_summary,
    )


def _sequential_finetune_stage1(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_checkpoints: bool,
    erm_hparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune stage1_lr, then weight_decay, then batch_size; others at grid middle.

    When ``erm_hparams`` is provided (Comp_models ERM finetune summary), the
    lr/weight_decay/batch_size grid search is skipped entirely: Stage 1 is trained
    once with ``cfg.stage1_lr``/``cfg.weight_decay``/``cfg.batch_size`` (already
    applied from the summary), still using worst_subgroup validation selection
    across epochs. Winning Stage-1 batch_size is reused for Stages 2 and 4.
    """
    if erm_hparams is not None:
        print("\n========== Stage 1: using ERM finetune hparams (grid search skipped) ==========")
        trial_cfg = copy.deepcopy(cfg)
        hparams = {
            "lr": trial_cfg.stage1_lr,
            "weight_decay": trial_cfg.weight_decay,
            "batch_size": trial_cfg.batch_size,
            "stage1_epochs": trial_cfg.stage1_epochs,
            "source": "erm_finetune_summary",
        }
        print_finetune_trial("Stage 1 (ERM hparams)", hparams)
        set_seed(cfg.seed)
        model = build_model(trial_cfg, device)
        model = train_stage1_baseline(model, train_loader, val_loader, device, trial_cfg)
        report = evaluate_head_report(model, val_loader, device, trial_cfg, head="baseline")
        metrics = report_selection_metrics(report, trial_cfg)
        best_state = state_dict_to_cpu(model)
        best_cfg = trial_cfg

        if save_checkpoints:
            ckpt_model = build_model(best_cfg, device)
            load_matching_state_dict(ckpt_model, best_state)
            save_stage_checkpoint(best_cfg, ckpt_model, "stage1")
            build_stage1_baseline_reference(best_cfg, ckpt_model, device, save=True)

        return {
            "stage1": {
                "hparams": hparams,
                "metrics": metrics,
                "split": "val",
            },
            "best_state": best_state,
            "best_cfg": best_cfg,
        }

    mid_wd = grid_middle(cfg.stage1_weight_decay_grid)
    mid_bs = int(grid_middle(cfg.stage1_batch_size_grid))
    best_lr = float(grid_middle(cfg.stage1_lr_grid))
    best_wd = float(mid_wd)
    best_bs = mid_bs
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_cfg: Config | None = None
    lr_trials: list[dict[str, Any]] = []

    def _run_stage1_trial(lr: float, weight_decay: float, batch_size: int) -> tuple[dict[str, float], dict[str, torch.Tensor], Config]:
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.stage1_lr = float(lr)
        trial_cfg.weight_decay = float(weight_decay)
        trial_cfg.batch_size = int(batch_size)
        trial_train, trial_val, _ = build_dataloaders(trial_cfg)
        set_seed(cfg.seed)
        model = build_model(trial_cfg, device)
        model = train_stage1_baseline(model, trial_train, trial_val, device, trial_cfg)
        report = evaluate_head_report(model, trial_val, device, trial_cfg, head="baseline")
        metrics = report_selection_metrics(report, trial_cfg)
        return metrics, state_dict_to_cpu(model), trial_cfg

    print("\n========== Sequential finetune Stage 1: lr (wd/batch fixed at middle) ==========")
    for lr in cfg.stage1_lr_grid:
        hparams = {"lr": float(lr), "weight_decay": mid_wd, "batch_size": mid_bs}
        print_finetune_trial("Stage 1 lr", hparams)
        metrics, state, trial_cfg = _run_stage1_trial(float(lr), mid_wd, mid_bs)
        lr_trials.append({"hparams": hparams, "metrics": metrics})
        if is_better_metrics(
            metrics, best_metrics, trial_cfg.selection_metric_stage1, trial_cfg.use_overall_tie_break
        ):
            best_lr = float(lr)
            best_metrics = metrics
            best_state = state
            best_cfg = copy.deepcopy(trial_cfg)

    if best_state is None or best_cfg is None or best_metrics is None:
        raise RuntimeError("Stage 1 sequential finetune (lr) failed.")

    wd_trials: list[dict[str, Any]] = []
    print("\n========== Sequential finetune Stage 1: weight_decay (lr/batch fixed) ==========")
    for wd in cfg.stage1_weight_decay_grid:
        hparams = {"lr": best_lr, "weight_decay": float(wd), "batch_size": mid_bs}
        print_finetune_trial("Stage 1 wd", hparams)
        metrics, state, trial_cfg = _run_stage1_trial(best_lr, float(wd), mid_bs)
        wd_trials.append({"hparams": hparams, "metrics": metrics})
        if is_better_metrics(
            metrics, best_metrics, trial_cfg.selection_metric_stage1, trial_cfg.use_overall_tie_break
        ):
            best_wd = float(wd)
            best_metrics = metrics
            best_state = state
            best_cfg = copy.deepcopy(trial_cfg)

    bs_trials: list[dict[str, Any]] = []
    print("\n========== Sequential finetune Stage 1: batch_size (lr/wd fixed at best) ==========")
    for bs in cfg.stage1_batch_size_grid:
        hparams = {"lr": best_lr, "weight_decay": best_wd, "batch_size": int(bs)}
        print_finetune_trial("Stage 1 batch_size", hparams)
        metrics, state, trial_cfg = _run_stage1_trial(best_lr, best_wd, int(bs))
        bs_trials.append({"hparams": hparams, "metrics": metrics})
        if is_better_metrics(
            metrics, best_metrics, trial_cfg.selection_metric_stage1, trial_cfg.use_overall_tie_break
        ):
            best_bs = int(bs)
            best_metrics = metrics
            best_state = state
            best_cfg = copy.deepcopy(trial_cfg)

    if save_checkpoints and best_cfg is not None and best_state is not None:
        model = build_model(best_cfg, device)
        load_matching_state_dict(model, best_state)
        save_stage_checkpoint(best_cfg, model, "stage1")
        build_stage1_baseline_reference(best_cfg, model, device, save=True)

    return {
        "stage1": {
            "hparams": {
                "lr": best_lr,
                "weight_decay": best_wd,
                "batch_size": best_bs,
                "stage1_epochs": cfg.stage1_epochs,
            },
            "metrics": best_metrics,
            "split": "val",
            "lr_trials": lr_trials,
            "weight_decay_trials": wd_trials,
            "batch_size_trials": bs_trials,
        },
        "best_state": best_state,
        "best_cfg": best_cfg,
    }


def _sequential_finetune_stage2(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage1_ref: Stage1BaselineReference,
    stage1_baseline_val_report: dict[str, float],
    stage1_state: dict[str, torch.Tensor],
    save_checkpoints: bool,
) -> dict[str, Any]:
    """Tune demo_embed_dim only (Stage 2)."""
    best_state: dict[str, torch.Tensor] | None = None
    best_cfg: Config | None = None
    best_metrics: dict[str, float] | None = None
    best_hparams: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []

    print("\n========== Sequential finetune Stage 2: demo_embed_dim ==========")
    for demo_embed_dim in cfg.stage2_demo_embed_dim_grid:
        hparams = {"demo_embed_dim": int(demo_embed_dim), "stage2_epochs": cfg.stage2_epochs}
        print_finetune_trial("Stage 2", hparams)
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.demo_embed_dim = int(demo_embed_dim)
        set_seed(cfg.seed)
        model = build_model(trial_cfg, device)
        load_matching_state_dict(model, stage1_state)
        model = train_stage2_demo(model, train_loader, val_loader, device, trial_cfg, stage1_ref)
        demo_report = evaluate_head_report(model, val_loader, device, trial_cfg, head="demo")
        metrics = report_selection_metrics(
            demo_report, trial_cfg, improvement_base_report=stage1_baseline_val_report
        )
        trials.append({"hparams": hparams, "metrics": metrics})
        if is_better_metrics(
            metrics, best_metrics, trial_cfg.selection_metric_stage2, trial_cfg.use_overall_tie_break
        ):
            best_state = state_dict_to_cpu(model)
            best_cfg = copy.deepcopy(trial_cfg)
            best_metrics = metrics
            best_hparams = hparams

    if best_state is None or best_cfg is None or best_metrics is None or best_hparams is None:
        raise RuntimeError("Stage 2 sequential finetune failed.")

    if save_checkpoints:
        model = build_model(best_cfg, device)
        load_matching_state_dict(model, best_state)
        save_stage_checkpoint(best_cfg, model, "stage2")

    return {
        "stage2": {"hparams": best_hparams, "metrics": best_metrics, "split": "val", "trials": trials},
        "best_state": best_state,
        "best_cfg": best_cfg,
    }


def _sequential_finetune_stage3(
    cfg: Config,
    base_cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage1_ref: Stage1BaselineReference,
    stage1_baseline_val_report: dict[str, float],
    stage2_state: dict[str, torch.Tensor],
    demo_report: dict[str, float],
) -> dict[str, Any]:
    """Tune alpha then lambda_cv: compute usefulness, train Stage 4 (probe hparams), select by Stage-4 val."""
    selection_metric = effective_selection_metric("stage4", cfg.selection_metric_stage4)
    stage4_probe_hparams = _initial_stage4_hparams(cfg, use_grid_middles=True)
    mid_cv = grid_middle(cfg.stage3_lambda_cv_grid)
    best_alpha = grid_middle(cfg.stage3_alpha_grid)
    best_cv = mid_cv
    best_stage4_metrics: dict[str, float] | None = None
    best_usefulness_score = float("nan")
    best_usefulness_meta: dict[str, Any] | None = None
    alpha_trials: list[dict[str, Any]] = []

    print(
        "\n========== Sequential finetune Stage 3: alpha (lambda_cv at middle; "
        f"Stage-4 probe, select by {selection_metric}) =========="
    )
    print(f"Stage-4 probe hparams (fixed during Stage 3): {stage4_probe_hparams}")
    for alpha in cfg.stage3_alpha_grid:
        hparams = {"alpha": float(alpha), "lambda_cv": mid_cv}
        usefulness_score, usefulness_meta = compute_usefulness_score(
            stage1_baseline_val_report, demo_report, base_cfg.num_subgroups, float(alpha), mid_cv, base_cfg.eps
        )
        print_finetune_trial(
            "Stage 3 alpha",
            {**hparams, "usefulness_score": usefulness_score, "stage4_probe_hparams": stage4_probe_hparams},
        )
        _model, stage4_metrics, _stage4_hparams = _run_stage4_trial(
            base_cfg,
            stage2_state,
            train_loader,
            val_loader,
            device,
            stage1_ref,
            stage1_baseline_val_report,
            usefulness_score,
            stage4_probe_hparams,
            save_checkpoints=False,
        )
        trial_metrics = {
            "usefulness_score": usefulness_score,
            **usefulness_meta,
            **{k: v for k, v in stage4_metrics.items() if k != "usefulness_score"},
        }
        alpha_trials.append(
            {
                "hparams": hparams,
                "stage4_probe_hparams": dict(stage4_probe_hparams),
                "metrics": trial_metrics,
            }
        )
        if is_better_metrics(
            stage4_metrics, best_stage4_metrics, selection_metric, cfg.use_overall_tie_break
        ):
            best_alpha = float(alpha)
            best_stage4_metrics = stage4_metrics
            best_usefulness_score = usefulness_score
            best_usefulness_meta = usefulness_meta

    cv_trials: list[dict[str, Any]] = []
    print(
        "\n========== Sequential finetune Stage 3: lambda_cv (alpha at best; "
        f"Stage-4 probe, select by {selection_metric}) =========="
    )
    for lambda_cv in cfg.stage3_lambda_cv_grid:
        hparams = {"alpha": best_alpha, "lambda_cv": float(lambda_cv)}
        usefulness_score, usefulness_meta = compute_usefulness_score(
            stage1_baseline_val_report,
            demo_report,
            base_cfg.num_subgroups,
            best_alpha,
            float(lambda_cv),
            base_cfg.eps,
        )
        print_finetune_trial(
            "Stage 3 lambda_cv",
            {**hparams, "usefulness_score": usefulness_score, "stage4_probe_hparams": stage4_probe_hparams},
        )
        _model, stage4_metrics, _stage4_hparams = _run_stage4_trial(
            base_cfg,
            stage2_state,
            train_loader,
            val_loader,
            device,
            stage1_ref,
            stage1_baseline_val_report,
            usefulness_score,
            stage4_probe_hparams,
            save_checkpoints=False,
        )
        trial_metrics = {
            "usefulness_score": usefulness_score,
            **usefulness_meta,
            **{k: v for k, v in stage4_metrics.items() if k != "usefulness_score"},
        }
        cv_trials.append(
            {
                "hparams": hparams,
                "stage4_probe_hparams": dict(stage4_probe_hparams),
                "metrics": trial_metrics,
            }
        )
        if is_better_metrics(
            stage4_metrics, best_stage4_metrics, selection_metric, cfg.use_overall_tie_break
        ):
            best_cv = float(lambda_cv)
            best_stage4_metrics = stage4_metrics
            best_usefulness_score = usefulness_score
            best_usefulness_meta = usefulness_meta

    if best_stage4_metrics is None or best_usefulness_meta is None:
        raise RuntimeError("Stage 3 sequential finetune failed.")

    base_cfg.alpha = best_alpha
    base_cfg.lambda_cv = best_cv
    return {
        "stage3": {
            "hparams": {"alpha": best_alpha, "lambda_cv": best_cv},
            "metrics": {
                "usefulness_score": best_usefulness_score,
                **best_usefulness_meta,
                **{k: v for k, v in best_stage4_metrics.items() if k != "usefulness_score"},
            },
            "split": "val",
            "selection_metric_stage4": selection_metric,
            "stage4_probe_hparams": dict(stage4_probe_hparams),
            "alpha_trials": alpha_trials,
            "lambda_cv_trials": cv_trials,
        },
        "usefulness_score": best_usefulness_score,
        "usefulness_meta": best_usefulness_meta,
    }


def _merge_finetune_summary(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    merged.update(updates)
    return merged


def _write_finetune_outputs(
    cfg: Config,
    summary: dict[str, Any],
    test_results: dict[str, Any] | None = None,
) -> None:
    out_root = finetune_results_dir(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    final_dir = out_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if test_results is not None:
        summary["final"] = {"output_dir": str(final_dir), "test_results": test_results}
        with open(final_dir / "final_test_results.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable(test_results), f, indent=2)
    with open(out_root / "finetune_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print(f"\nSaved finetuning summary → {(out_root / 'finetune_summary.json').resolve()}")


def _finetune_stages_1_2_3_grids(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_checkpoints: bool,
    erm_hparams: dict[str, Any] | None = None,
) -> tuple[
    Stage1BaselineReference,
    dict[str, torch.Tensor],
    Config,
    dict[str, float],
    float,
    dict[str, Any],
    dict[str, Any],
]:
    """Stages 1–2 grid search + Stage 3 usefulness grid (same as joint finetune preamble).

    When ``erm_hparams`` is provided, Stage 1's lr/weight_decay grid search is
    skipped and Stage 1 is trained once with ``cfg.stage1_lr``/``cfg.weight_decay``
    (already applied from the Comp_models ERM finetune summary).
    """
    select_loader = val_loader

    best_stage1_state: dict[str, torch.Tensor] | None = None
    best_stage1_cfg: Config | None = None
    best_stage1_metrics: dict[str, float] | None = None
    best_stage1_hparams: dict[str, Any] | None = None

    if erm_hparams is not None:
        print("\n========== Finetune Stage 1: using ERM hparams (grid search skipped) ==========")
        hparams = {
            "lr": cfg.stage1_lr,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "stage1_epochs": cfg.stage1_epochs,
            "source": "erm_finetune_summary",
        }
        print_finetune_trial("Stage 1 (ERM hparams)", hparams)
        trial_cfg = copy.deepcopy(cfg)
        set_seed(cfg.seed)
        model = build_model(trial_cfg, device)
        model = train_stage1_baseline(model, train_loader, val_loader, device, trial_cfg)
        report = evaluate_head_report(model, select_loader, device, trial_cfg, head="baseline")
        metrics = report_selection_metrics(report, trial_cfg)
        best_stage1_state = state_dict_to_cpu(model)
        best_stage1_cfg = copy.deepcopy(trial_cfg)
        best_stage1_metrics = metrics
        best_stage1_hparams = hparams
    else:
        s1 = _sequential_finetune_stage1(
            cfg, train_loader, val_loader, device, save_checkpoints=False, erm_hparams=None
        )
        best_stage1_state = s1["best_state"]
        best_stage1_cfg = s1["best_cfg"]
        best_stage1_metrics = s1["stage1"]["metrics"]
        best_stage1_hparams = s1["stage1"]["hparams"]

    if best_stage1_state is None or best_stage1_cfg is None or best_stage1_metrics is None or best_stage1_hparams is None:
        raise RuntimeError("Stage 1 finetuning did not produce a valid checkpoint.")

    cfg.batch_size = int(best_stage1_cfg.batch_size)
    train_loader, val_loader, _ = build_dataloaders(best_stage1_cfg)
    select_loader = val_loader

    if save_checkpoints:
        stage1_model = build_model(best_stage1_cfg, device)
        load_matching_state_dict(stage1_model, best_stage1_state)
        save_stage_checkpoint(best_stage1_cfg, stage1_model, "stage1")

    stage1_ref_model = build_model(best_stage1_cfg, device)
    load_matching_state_dict(stage1_ref_model, best_stage1_state)
    stage1_ref = build_stage1_baseline_reference(best_stage1_cfg, stage1_ref_model, device, save=save_checkpoints)
    stage1_baseline_val_report = stage1_ref.auc_report(select_loader, device, best_stage1_cfg.num_subgroups)

    best_stage2_state: dict[str, torch.Tensor] | None = None
    best_stage2_cfg: Config | None = None
    best_stage2_metrics: dict[str, float] | None = None
    best_stage2_hparams: dict[str, Any] | None = None
    best_stage2_demo_report: dict[str, float] | None = None

    print("\n========== Finetune Stage 2 ==========")
    for demo_embed_dim in cfg.stage2_demo_embed_dim_grid:
        hparams = {"demo_embed_dim": demo_embed_dim, "stage2_epochs": best_stage1_cfg.stage2_epochs}
        print_finetune_trial("Stage 2", hparams)
        trial_cfg = copy.deepcopy(best_stage1_cfg)
        trial_cfg.demo_embed_dim = demo_embed_dim
        set_seed(cfg.seed)
        model = build_model(trial_cfg, device)
        load_matching_state_dict(model, best_stage1_state)
        model = train_stage2_demo(model, train_loader, val_loader, device, trial_cfg, stage1_ref)
        demo_report = evaluate_head_report(model, select_loader, device, trial_cfg, head="demo")
        metrics = report_selection_metrics(
            demo_report, trial_cfg, improvement_base_report=stage1_baseline_val_report
        )
        if is_better_metrics(
            metrics, best_stage2_metrics, trial_cfg.selection_metric_stage2, trial_cfg.use_overall_tie_break
        ):
            best_stage2_state = state_dict_to_cpu(model)
            best_stage2_cfg = copy.deepcopy(trial_cfg)
            best_stage2_metrics = metrics
            best_stage2_hparams = hparams
            best_stage2_demo_report = demo_report

    if (
        best_stage2_state is None
        or best_stage2_cfg is None
        or best_stage2_metrics is None
        or best_stage2_hparams is None
        or best_stage2_demo_report is None
    ):
        raise RuntimeError("Stage 2 finetuning did not produce a valid checkpoint.")

    if save_checkpoints:
        stage2_model = build_model(best_stage2_cfg, device)
        load_matching_state_dict(stage2_model, best_stage2_state)
        save_stage_checkpoint(best_stage2_cfg, stage2_model, "stage2")

    stage3_result = _sequential_finetune_stage3(
        cfg,
        best_stage2_cfg,
        train_loader,
        val_loader,
        device,
        stage1_ref,
        stage1_baseline_val_report,
        best_stage2_state,
        best_stage2_demo_report,
    )
    best_usefulness_score = stage3_result["usefulness_score"]
    best_usefulness_meta = stage3_result["usefulness_meta"]

    early_summary = {
        "stage1": {"hparams": best_stage1_hparams, "metrics": best_stage1_metrics, "split": "val"},
        "stage2": {"hparams": best_stage2_hparams, "metrics": best_stage2_metrics, "split": "val"},
        "stage3": stage3_result["stage3"],
    }

    return (
        stage1_ref,
        best_stage2_state,
        best_stage2_cfg,
        stage1_baseline_val_report,
        best_usefulness_score,
        best_usefulness_meta,
        early_summary,
        train_loader,
        val_loader,
    )


def _run_stage4_trial(
    base_cfg: Config,
    stage2_state: dict[str, torch.Tensor],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage1_ref: Stage1BaselineReference,
    stage1_baseline_val_report: dict[str, float],
    usefulness_score: float,
    stage4_hparams: dict[str, Any],
    save_checkpoints: bool,
) -> tuple[FilmResidualFairnessModel, dict[str, float], dict[str, Any]]:
    trial_cfg = copy.deepcopy(base_cfg)
    trial_cfg.lambda_imp = float(stage4_hparams["lambda_imp"])
    trial_cfg.lambda_worst = float(stage4_hparams["lambda_worst"])
    trial_cfg.lambda_kd = float(stage4_hparams["lambda_kd"])
    trial_cfg.temperature = float(stage4_hparams.get("temperature", base_cfg.temperature))
    trial_cfg.gamma = float(stage4_hparams["gamma"])
    trial_cfg.stage4_head_lr = float(stage4_hparams["stage4_head_lr"])
    trial_cfg.stage4_encoder_lr = float(stage4_hparams["stage4_encoder_lr"])
    trial_cfg.stage4_epochs = int(stage4_hparams.get("stage4_epochs", base_cfg.stage4_epochs))
    trial_cfg.stage4_unfreeze_from_block = int(stage4_hparams["stage4_unfreeze_from_block"])
    set_seed(base_cfg.seed)
    model = build_model(trial_cfg, device)
    load_matching_state_dict(model, stage2_state)
    model, bundle, _paths = train_stage4_proposed(
        model,
        train_loader,
        val_loader,
        device,
        trial_cfg,
        usefulness_score=usefulness_score,
        stage1_ref=stage1_ref,
        save_checkpoints=save_checkpoints,
    )
    # train_stage4_proposed leaves `model` at its last-epoch state, not necessarily the
    # best one; load the val-selected checkpoint (selection_metric_stage4) before scoring.
    trial_selection_metric = effective_selection_metric("stage4", trial_cfg.selection_metric_stage4)
    model.load_state_dict(bundle[trial_selection_metric]["state"])
    proposed_report = evaluate_head_report(
        model, val_loader, device, trial_cfg, head="proposed", usefulness_score=usefulness_score
    )
    metrics = {
        "usefulness_score": usefulness_score,
        **report_selection_metrics(
            proposed_report, trial_cfg, improvement_base_report=stage1_baseline_val_report
        ),
    }
    hparams = {
        "lambda_imp": trial_cfg.lambda_imp,
        "lambda_worst": trial_cfg.lambda_worst,
        "lambda_kd": trial_cfg.lambda_kd,
        "temperature": trial_cfg.temperature,
        "gamma": trial_cfg.gamma,
        "stage4_head_lr": trial_cfg.stage4_head_lr,
        "stage4_encoder_lr": trial_cfg.stage4_encoder_lr,
        "stage4_epochs": trial_cfg.stage4_epochs,
        "stage4_unfreeze_from_block": trial_cfg.stage4_unfreeze_from_block,
    }
    return model, metrics, hparams


def _sequential_tune_stage4_param(
    param_name: str,
    grid: tuple[Any, ...],
    fixed: dict[str, Any],
    base_cfg: Config,
    stage2_state: dict[str, torch.Tensor],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage1_ref: Stage1BaselineReference,
    stage1_baseline_val_report: dict[str, float],
    usefulness_score: float,
    selection_metric: str,
    use_overall_tie_break: bool,
    save_checkpoints: bool,
) -> tuple[Any, dict[str, float], dict[str, Any], list[dict[str, Any]], FilmResidualFairnessModel]:
    stage4_hparams = dict(fixed)
    is_int = param_name in STAGE4_INT_HPARAMS
    best_value: Any = grid_middle_int(grid) if is_int else float(grid_middle(grid))  # type: ignore[arg-type]
    best_metrics: dict[str, float] | None = None
    best_hparams: dict[str, Any] | None = None
    best_model: FilmResidualFairnessModel | None = None
    trial_records: list[dict[str, Any]] = []

    print(f"\n========== Sequential finetune Stage 4: {param_name} (grid={list(grid)}) ==========")
    print(f"Fixed while tuning: { {k: v for k, v in stage4_hparams.items() if k != param_name} }")

    for trial_idx, value in enumerate(grid, start=1):
        coerced = int(value) if is_int else float(value)
        stage4_hparams[param_name] = coerced
        print_finetune_trial(
            f"Stage 4 [{param_name}] [{trial_idx}/{len(grid)}]",
            {k: stage4_hparams[k] for k in stage4_hparams},
        )
        model, metrics, full_hparams = _run_stage4_trial(
            base_cfg,
            stage2_state,
            train_loader,
            val_loader,
            device,
            stage1_ref,
            stage1_baseline_val_report,
            usefulness_score,
            stage4_hparams,
            save_checkpoints,
        )
        trial_records.append({"hparams": full_hparams, "metrics": metrics})
        if is_better_metrics(metrics, best_metrics, selection_metric, use_overall_tie_break):
            best_value = coerced
            best_metrics = metrics
            best_hparams = full_hparams
            best_model = model

    if best_metrics is None or best_hparams is None or best_model is None:
        raise RuntimeError(f"Sequential finetune for {param_name} did not produce a valid trial.")

    print(f"Best {param_name}={best_value} ({selection_metric}={_selection_primary(best_metrics, selection_metric):.4f})")
    return best_value, best_metrics, best_hparams, trial_records, best_model


def _sequential_finetune_stage4(
    cfg: Config,
    base_cfg: Config,
    stage2_state: dict[str, torch.Tensor],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    stage1_ref: Stage1BaselineReference,
    stage1_baseline_val_report: dict[str, float],
    usefulness_score: float,
    *,
    use_grid_middles: bool,
    save_checkpoints: bool,
) -> tuple[dict[str, Any], dict[str, Any], FilmResidualFairnessModel]:
    """Sequentially tune all Stage-4 hyperparameters; return summary blocks + best model."""
    selection_metric = effective_selection_metric("stage4", cfg.selection_metric_stage4)
    stage4_hparams = _initial_stage4_hparams(base_cfg, use_grid_middles=use_grid_middles)
    stage4_summary_blocks: dict[str, Any] = {"grid_middles": dict(stage4_hparams)}
    best_model: FilmResidualFairnessModel | None = None
    best_metrics: dict[str, float] | None = None
    best_hparams: dict[str, Any] | None = None

    for param_name, grid_attr in STAGE4_SEQUENTIAL_TUNING:
        grid = getattr(cfg, grid_attr)
        fixed_snapshot = {k: v for k, v in stage4_hparams.items() if k != param_name}
        best_val, metrics, hparams, trials, model = _sequential_tune_stage4_param(
            param_name,
            grid,
            stage4_hparams,
            base_cfg,
            stage2_state,
            train_loader,
            val_loader,
            device,
            stage1_ref,
            stage1_baseline_val_report,
            usefulness_score,
            selection_metric,
            cfg.use_overall_tie_break,
            save_checkpoints,
        )
        stage4_hparams[param_name] = best_val
        stage4_summary_blocks[_stage4_summary_key(param_name)] = {
            "fixed": fixed_snapshot,
            "trials": trials,
            "best_value": best_val,
            "best_metrics": metrics,
        }
        best_model = model
        best_metrics = metrics
        best_hparams = hparams

    if best_model is None or best_metrics is None or best_hparams is None:
        raise RuntimeError("Stage 4 sequential finetune did not produce a valid model.")

    stage4_summary_blocks["stage4"] = {
        "hparams": best_hparams,
        "metrics": best_metrics,
        "split": "val",
    }
    stage4_summary_blocks["selection_metric_stage4"] = selection_metric
    return stage4_summary_blocks, best_hparams, best_model


def run_sequential_lambda_finetuning_pipeline(cfg: Config) -> None:
    """Modular sequential finetune: one hyperparameter at a time; skip stages loaded from summary."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    save_ckpt = cfg.save_finetune_checkpoints
    only_stage = cfg.finetune_only_stage.strip().lower()
    if only_stage not in FINETUNE_ONLY_STAGES:
        raise ValueError(f"finetune_only_stage must be one of {FINETUNE_ONLY_STAGES}, got {only_stage!r}")

    erm_hparams: dict[str, Any] | None = None
    if only_stage in ("all", "stage1"):
        erm_hparams = load_erm_stage1_hparams(cfg, backbone=normalize_backbone(cfg.backbone))
        if erm_hparams is not None:
            apply_erm_hparams_to_cfg(cfg, erm_hparams)

    print("========== Modular Sequential Finetuning (model_finalizing) ==========")
    print("Device:", device)
    print("Run name:", cfg.run_name)
    print("Finetune only stage:", only_stage)
    print("Save checkpoints:", save_ckpt)

    base_summary = read_finetune_summary(cfg.finetune_from_summary) if cfg.finetune_from_summary else {}
    if only_stage in ("stage2", "stage3", "stage4") and base_summary:
        apply_summary_hparams_to_cfg(cfg, base_summary)
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    if only_stage == "stage1":
        result = _sequential_finetune_stage1(
            cfg, train_loader, val_loader, device, save_ckpt, erm_hparams=erm_hparams
        )
        summary = _merge_finetune_summary(
            base_summary,
            {"finetune_mode": "sequential_stage1", "stage1": result["stage1"], "selection_split": "val"},
        )
        _write_finetune_outputs(cfg, summary)
        return

    if only_stage == "stage2":
        if not cfg.finetune_from_summary:
            raise ValueError("Stage 2-only finetune requires --finetune-from-summary with completed Stage 1.")
        apply_summary_hparams_to_cfg(cfg, base_summary)
        stage1_ref = Stage1BaselineReference.from_checkpoint(cfg, device)
        stage1_state = load_checkpoint_state(scratch_checkpoint_dir(cfg, "stage1") / "checkpoint.pt")
        stage1_baseline_val_report = stage1_ref.auc_report(val_loader, device, cfg.num_subgroups)
        result = _sequential_finetune_stage2(
            cfg, train_loader, val_loader, device, stage1_ref, stage1_baseline_val_report, stage1_state, save_ckpt
        )
        summary = _merge_finetune_summary(
            base_summary,
            {"finetune_mode": "sequential_stage2", "stage2": result["stage2"], "selection_split": "val"},
        )
        _write_finetune_outputs(cfg, summary)
        return

    if only_stage == "stage3":
        if not cfg.finetune_from_summary:
            raise ValueError("Stage 3-only finetune requires --finetune-from-summary with Stages 1–2 done.")
        apply_summary_hparams_to_cfg(cfg, base_summary)
        stage1_ref = Stage1BaselineReference.from_checkpoint(cfg, device)
        stage2_state = load_stage2_state_from_checkpoint(cfg)
        work_cfg = copy.deepcopy(cfg)
        stage1_baseline_val_report = stage1_ref.auc_report(val_loader, device, work_cfg.num_subgroups)
        stage2_model = build_model(work_cfg, device)
        load_matching_state_dict(stage2_model, stage2_state)
        demo_report = evaluate_head_report(stage2_model, val_loader, device, work_cfg, head="demo")
        result = _sequential_finetune_stage3(
            cfg,
            work_cfg,
            train_loader,
            val_loader,
            device,
            stage1_ref,
            stage1_baseline_val_report,
            stage2_state,
            demo_report,
        )
        summary = _merge_finetune_summary(
            base_summary,
            {
                "finetune_mode": "sequential_stage3",
                "stage3": result["stage3"],
                "selection_split": "val",
            },
        )
        _write_finetune_outputs(cfg, summary)
        return

    if only_stage == "stage4":
        if not cfg.finetune_from_summary:
            raise ValueError(
                "Stage 4-only finetune requires --finetune-from-summary and on-disk Stage 1–2 checkpoints."
            )
        (
            stage1_ref,
            stage2_state,
            base_cfg,
            stage1_baseline_val_report,
            usefulness_score,
            _usefulness_meta,
            early_summary,
        ) = _load_stages_through_3_from_artifacts(cfg, val_loader, device, base_summary)
    elif only_stage == "all":
        (
            stage1_ref,
            stage2_state,
            base_cfg,
            stage1_baseline_val_report,
            usefulness_score,
            _usefulness_meta,
            early_summary,
            train_loader,
            val_loader,
        ) = _finetune_stages_1_2_3_grids(
            cfg, train_loader, val_loader, device, save_ckpt, erm_hparams=erm_hparams
        )
        cfg.batch_size = int(base_cfg.batch_size)
    else:
        raise RuntimeError(f"Unhandled finetune_only_stage: {only_stage}")

    n_stage4_trials = sum(len(getattr(cfg, grid_attr)) for _param, grid_attr in STAGE4_SEQUENTIAL_TUNING)
    print(
        f"\nStage 4 sequential finetune: {n_stage4_trials} trainings across "
        f"{len(STAGE4_SEQUENTIAL_TUNING)} hyperparameters"
    )

    sequential_summary: dict[str, Any] = _merge_finetune_summary(
        base_summary if only_stage == "stage4" else {},
        {
            "finetune_mode": "sequential_lambdas" if only_stage == "all" else "sequential_stage4",
            "finetune_only_stage": only_stage,
            **early_summary,
        },
    )

    stage4_blocks, _best_hparams, best_model = _sequential_finetune_stage4(
        cfg,
        base_cfg,
        stage2_state,
        train_loader,
        val_loader,
        device,
        stage1_ref,
        stage1_baseline_val_report,
        usefulness_score,
        use_grid_middles=(only_stage == "all"),
        save_checkpoints=save_ckpt,
    )
    sequential_summary.update(stage4_blocks)
    sequential_summary["selection_split"] = "val"

    test_results = final_evaluation(
        best_model, test_loader, device, base_cfg, usefulness_score, stage1_ref
    )
    _write_finetune_outputs(cfg, sequential_summary, test_results)
    print("Final Stage-4 hparams:", _best_hparams)


def run_finetuning_pipeline(cfg: Config) -> None:
    """Grid-search hyperparameters per stage on cfg.seed; write finetune_summary.json.

    Trial selection uses validation-set metrics. Only the final best model checkpoint
    is written under finetune/best/checkpoint.pt. Test is evaluated once at the end.
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)

    print("========== Staged Hyperparameter Finetuning (model_finalizing) ==========")
    print("Device:", device)
    print("Run name:", cfg.run_name)
    print("Fusion:", cfg.fusion_type)
    print("Finetune seed:", cfg.seed)
    print("Hyperparameter selection split: validation (epoch checkpoints within each trial also use val)")

    erm_hparams = load_erm_stage1_hparams(cfg, backbone=normalize_backbone(cfg.backbone))
    if erm_hparams is not None:
        apply_erm_hparams_to_cfg(cfg, erm_hparams)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    select_loader = val_loader
    out_root = finetune_results_dir(cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    # --- Stage 1 ---
    best_stage1_state: dict[str, torch.Tensor] | None = None
    best_stage1_cfg: Config | None = None
    best_stage1_metrics: dict[str, float] | None = None
    best_stage1_hparams: dict[str, Any] | None = None

    if erm_hparams is not None:
        print("\n========== Finetune Stage 1: using ERM hparams (grid search skipped) ==========")
        hparams = {
            "lr": cfg.stage1_lr,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "stage1_epochs": cfg.stage1_epochs,
            "source": "erm_finetune_summary",
        }
        print_finetune_trial("Stage 1 (ERM hparams)", hparams)
        trial_cfg = copy.deepcopy(cfg)
        set_seed(cfg.seed)
        model = build_model(trial_cfg, device)
        model = train_stage1_baseline(model, train_loader, val_loader, device, trial_cfg)
        report = evaluate_head_report(model, select_loader, device, trial_cfg, head="baseline")
        metrics = report_selection_metrics(report, trial_cfg)
        best_stage1_state = state_dict_to_cpu(model)
        best_stage1_cfg = copy.deepcopy(trial_cfg)
        best_stage1_metrics = metrics
        best_stage1_hparams = hparams
    else:
        s1 = _sequential_finetune_stage1(
            cfg, train_loader, val_loader, device, save_checkpoints=False, erm_hparams=None
        )
        best_stage1_state = s1["best_state"]
        best_stage1_cfg = s1["best_cfg"]
        best_stage1_metrics = s1["stage1"]["metrics"]
        best_stage1_hparams = s1["stage1"]["hparams"]

    if best_stage1_state is None or best_stage1_cfg is None or best_stage1_metrics is None or best_stage1_hparams is None:
        raise RuntimeError("Stage 1 finetuning did not produce a valid checkpoint.")
    summary["stage1"] = {"hparams": best_stage1_hparams, "metrics": best_stage1_metrics, "split": "val"}

    cfg.batch_size = int(best_stage1_cfg.batch_size)
    train_loader, val_loader, test_loader = build_dataloaders(best_stage1_cfg)
    select_loader = val_loader

    stage1_ref_model = build_model(best_stage1_cfg, device)
    load_matching_state_dict(stage1_ref_model, best_stage1_state)
    stage1_ref = build_stage1_baseline_reference(best_stage1_cfg, stage1_ref_model, device)
    stage1_baseline_val_report = stage1_ref.auc_report(select_loader, device, best_stage1_cfg.num_subgroups)

    # --- Stage 2 ---
    best_stage2_state: dict[str, torch.Tensor] | None = None
    best_stage2_cfg: Config | None = None
    best_stage2_metrics: dict[str, float] | None = None
    best_stage2_hparams: dict[str, Any] | None = None
    best_stage2_demo_report: dict[str, float] | None = None

    print("\n========== Finetune Stage 2 ==========")
    for demo_embed_dim in cfg.stage2_demo_embed_dim_grid:
        hparams = {"demo_embed_dim": demo_embed_dim, "stage2_epochs": best_stage1_cfg.stage2_epochs}
        print_finetune_trial("Stage 2", hparams)
        trial_cfg = copy.deepcopy(best_stage1_cfg)
        trial_cfg.demo_embed_dim = demo_embed_dim
        set_seed(cfg.seed)
        model = build_model(trial_cfg, device)
        load_matching_state_dict(model, best_stage1_state)
        model = train_stage2_demo(model, train_loader, val_loader, device, trial_cfg, stage1_ref)
        demo_report = evaluate_head_report(model, select_loader, device, trial_cfg, head="demo")
        metrics = report_selection_metrics(
            demo_report, trial_cfg, improvement_base_report=stage1_baseline_val_report
        )
        if is_better_metrics(metrics, best_stage2_metrics, trial_cfg.selection_metric_stage2, trial_cfg.use_overall_tie_break):
            best_stage2_state = state_dict_to_cpu(model)
            best_stage2_cfg = copy.deepcopy(trial_cfg)
            best_stage2_metrics = metrics
            best_stage2_hparams = hparams
            best_stage2_demo_report = demo_report

    if (
        best_stage2_state is None or best_stage2_cfg is None or best_stage2_metrics is None
        or best_stage2_hparams is None or best_stage2_demo_report is None
    ):
        raise RuntimeError("Stage 2 finetuning did not produce a valid checkpoint.")
    summary["stage2"] = {"hparams": best_stage2_hparams, "metrics": best_stage2_metrics, "split": "val"}

    # --- Stage 3+4 joint: usefulness (lambda_cv, alpha) + Stage-4 loss (imp, worst, kd, gamma) ---
    n_stage34 = (
        len(cfg.stage3_lambda_cv_grid)
        * len(cfg.stage3_alpha_grid)
        * len(cfg.stage4_lambda_imp_grid)
        * len(cfg.stage4_lambda_worst_grid)
        * len(cfg.stage4_lambda_kd_grid)
        * len(cfg.stage4_gamma_grid)
        * len(cfg.stage4_head_lr_grid)
        * len(cfg.stage4_encoder_lr_grid)
    )
    print(
        f"\n========== Finetune Stage 3+4 joint ({n_stage34} Stage-4 trainings) =========="
    )
    print(
        "Grids: lambda_cv×alpha × imp×worst×kd×gamma×head_lr×enc_lr = "
        f"{len(cfg.stage3_lambda_cv_grid)}×{len(cfg.stage3_alpha_grid)} × "
        f"{len(cfg.stage4_lambda_imp_grid)}×{len(cfg.stage4_lambda_worst_grid)}×"
        f"{len(cfg.stage4_lambda_kd_grid)}×{len(cfg.stage4_gamma_grid)}×"
        f"{len(cfg.stage4_head_lr_grid)}×{len(cfg.stage4_encoder_lr_grid)}"
    )

    best_stage4_model: FilmResidualFairnessModel | None = None
    best_stage4_cfg: Config | None = None
    best_stage4_metrics: dict[str, float] | None = None
    best_stage4_hparams: dict[str, Any] | None = None
    best_stage4_report: dict[str, float] | None = None
    best_stage4_checkpoint_bundle: dict[str, dict[str, Any]] | None = None
    best_usefulness_score = float("nan")
    best_usefulness_meta: dict[str, Any] | None = None
    trial_idx = 0

    for lambda_cv in cfg.stage3_lambda_cv_grid:
        for alpha in cfg.stage3_alpha_grid:
            usefulness_score, usefulness_meta = compute_usefulness_score(
                stage1_baseline_val_report,
                best_stage2_demo_report,
                best_stage2_cfg.num_subgroups,
                alpha,
                lambda_cv,
                best_stage2_cfg.eps,
            )
            for lambda_imp in cfg.stage4_lambda_imp_grid:
                for lambda_worst in cfg.stage4_lambda_worst_grid:
                    for lambda_kd in cfg.stage4_lambda_kd_grid:
                        for gamma in cfg.stage4_gamma_grid:
                            for stage4_head_lr in cfg.stage4_head_lr_grid:
                                for stage4_encoder_lr in cfg.stage4_encoder_lr_grid:
                                    trial_idx += 1
                                    trial_cfg = copy.deepcopy(best_stage2_cfg)
                                    trial_cfg.lambda_cv = lambda_cv
                                    trial_cfg.alpha = alpha
                                    trial_cfg.lambda_imp = lambda_imp
                                    trial_cfg.lambda_worst = lambda_worst
                                    trial_cfg.lambda_kd = lambda_kd
                                    trial_cfg.gamma = gamma
                                    trial_cfg.stage4_head_lr = stage4_head_lr
                                    trial_cfg.stage4_encoder_lr = stage4_encoder_lr
                                    hparams = {
                                        "lambda_cv": lambda_cv,
                                        "alpha": alpha,
                                        "lambda_imp": lambda_imp,
                                        "lambda_worst": lambda_worst,
                                        "lambda_kd": lambda_kd,
                                        "gamma": gamma,
                                        "stage4_head_lr": stage4_head_lr,
                                        "stage4_encoder_lr": stage4_encoder_lr,
                                        "stage4_epochs": trial_cfg.stage4_epochs,
                                    }
                                    print_finetune_trial(f"Stage 3+4 [{trial_idx}/{n_stage34}]", hparams)
                                    set_seed(cfg.seed)
                                    model = build_model(trial_cfg, device)
                                    load_matching_state_dict(model, best_stage2_state)
                                    model, stage4_bundle, _stage4_paths = train_stage4_proposed(
                                        model,
                                        train_loader,
                                        val_loader,
                                        device,
                                        trial_cfg,
                                        usefulness_score=usefulness_score,
                                        stage1_ref=stage1_ref,
                                        save_checkpoints=False,
                                    )
                                    proposed_report = evaluate_head_report(
                                        model,
                                        select_loader,
                                        device,
                                        trial_cfg,
                                        head="proposed",
                                        usefulness_score=usefulness_score,
                                    )
                                    metrics = {
                                        "usefulness_score": usefulness_score,
                                        **report_selection_metrics(
                                            proposed_report,
                                            trial_cfg,
                                            improvement_base_report=stage1_baseline_val_report,
                                        ),
                                    }
                                    if is_better_metrics(
                                        metrics,
                                        best_stage4_metrics,
                                        trial_cfg.selection_metric_stage4,
                                        trial_cfg.use_overall_tie_break,
                                    ):
                                        best_stage4_model = model
                                        best_stage4_cfg = copy.deepcopy(trial_cfg)
                                        best_stage4_metrics = metrics
                                        best_stage4_hparams = hparams
                                        best_stage4_report = proposed_report
                                        best_stage4_checkpoint_bundle = copy.deepcopy(stage4_bundle)
                                        best_usefulness_score = usefulness_score
                                        best_usefulness_meta = usefulness_meta

    if (
        best_stage4_model is None
        or best_stage4_cfg is None
        or best_stage4_metrics is None
        or best_stage4_hparams is None
        or best_stage4_report is None
        or best_usefulness_meta is None
        or best_stage4_checkpoint_bundle is None
    ):
        raise RuntimeError("Stage 3+4 joint finetuning did not produce a valid checkpoint.")

    summary["stage3"] = {
        "hparams": {
            "lambda_cv": best_stage4_hparams["lambda_cv"],
            "alpha": best_stage4_hparams["alpha"],
        },
        "metrics": {
            "usefulness_score": best_usefulness_score,
            **best_usefulness_meta,
            **{k: v for k, v in best_stage4_metrics.items() if k != "usefulness_score"},
        },
        "split": "val",
    }

    test_results = evaluate_all_stage4_checkpoints(
        best_stage4_model,
        best_stage4_checkpoint_bundle,
        test_loader,
        device,
        best_stage4_cfg,
        best_usefulness_score,
        stage1_ref,
    )
    selected_metric = test_results["_selection"]["selected_checkpoint_metric"]
    final_dir = out_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    finetune_ckpt_paths = save_stage4_checkpoints(
        best_stage4_cfg, best_stage4_checkpoint_bundle, base_dir=final_dir / "checkpoints"
    )
    summary["stage4"] = {"hparams": best_stage4_hparams, "metrics": best_stage4_metrics, "split": "val"}
    summary["selection_split"] = "val"
    summary["stage34_joint_trials"] = n_stage34
    summary["final"] = {
        "output_dir": str(final_dir),
        "test_predictions_csv": str(final_dir / "test_predictions.csv"),
        "test_results": test_results,
        "stage4_checkpoints": finetune_ckpt_paths,
        "selected_checkpoint_metric": selected_metric,
        "final_selection_criterion": "test_min_subgroup_improvement",
    }

    best_ckpt_dir = out_root / "best"
    best_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = best_ckpt_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": {
                k: v.detach().cpu().clone()
                for k, v in best_stage4_checkpoint_bundle[selected_metric]["state"].items()
            },
            "hparams": best_stage4_hparams,
            "config": asdict(best_stage4_cfg),
            "finetune_seed": cfg.seed,
            "selection_split": "test",
            "selection_metric": selected_metric,
            "selection_criterion": "test_min_subgroup_improvement",
            "test_results": test_results,
        },
        best_ckpt_path,
    )
    print(
        f"Saved best finetune checkpoint (test-selected {selected_metric}) → {best_ckpt_path.resolve()}"
    )
    save_stage4_checkpoints(best_stage4_cfg, best_stage4_checkpoint_bundle, base_dir=best_ckpt_dir / "by_metric")

    with open(out_root / "finetune_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    prediction_paths = export_all_stage4_predictions(
        best_stage4_model,
        best_stage4_checkpoint_bundle,
        best_stage4_cfg,
        device,
        final_dir,
        best_usefulness_score,
        selected_metric=selected_metric,
    )
    with open(final_dir / "final_test_results.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(test_results), f, indent=2)
    with open(final_dir / "test_predictions_paths.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(prediction_paths), f, indent=2)

    print(f"\nSaved finetuning summary → {(out_root / 'finetune_summary.json').resolve()}")
    print("Best hyperparameters:")
    for stage_key in ("stage1", "stage2", "stage3", "stage4"):
        print(f"  {stage_key}: {summary[stage_key]['hparams']}")


def run_pipeline_stages_1_to_3(cfg: Config) -> None:
    """Train Stages 1–3 and persist checkpoints + stage123_summary.json (no Stage 4)."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)

    print("========== Proposed Fairness Pipeline (Stages 1–3 only) ==========")
    print("Device:", device)
    print("Run name:", cfg.run_name)
    print("Config:", cfg)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg, device)

    model = train_stage1_baseline(model, train_loader, val_loader, device, cfg)
    save_stage_checkpoint(cfg, model, "stage1")
    export_phase0_erm_baseline(cfg, model, device, val_loader, test_loader, seed=cfg.seed)
    stage1_ref = build_stage1_baseline_reference(cfg, model, device)

    model = train_stage2_demo(model, train_loader, val_loader, device, cfg, stage1_ref)
    save_stage_checkpoint(cfg, model, "stage2")
    usefulness_score, usefulness_meta, baseline_val, demo_val = stage3_compute_usefulness(
        model, val_loader, device, cfg, stage1_ref
    )
    save_stage123_summary(cfg, usefulness_score, usefulness_meta, baseline_val, demo_val)
    all_results = {
        "stopped_after": "stage3",
        "usefulness_score": usefulness_score,
        "usefulness_meta": usefulness_meta,
        "baseline_val": baseline_val,
        "demo_val": demo_val,
    }
    save_run_results(cfg, all_results)
    print("Stopped after Stage 3 (Stage 4 skipped).")


def run_pipeline_stage4_only(cfg: Config) -> None:
    """Load Stages 1–3 from --init-from-run-name, train Stage 4 under current run_name."""
    if not cfg.init_from_run_name:
        raise ValueError("--stage4-only requires --init-from-run-name pointing at a Stages 1–3 base run.")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    init_cfg = cfg_for_run_name(cfg, cfg.init_from_run_name)

    print("========== Proposed Fairness Pipeline (Stage 4 only) ==========")
    print("Device:", device)
    print("Run name (Stage 4 outputs):", cfg.run_name)
    print("Init from (Stages 1–3):", init_cfg.run_name)
    print(
        f"Stage 4 lambdas: λ_imp={cfg.lambda_imp} λ_worst={cfg.lambda_worst} "
        f"λ_kd={cfg.lambda_kd} γ={cfg.gamma}"
    )

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    (
        model,
        stage1_ref,
        usefulness_score,
        usefulness_meta,
        baseline_val,
        demo_val,
    ) = load_stage123_for_stage4(cfg, init_cfg, val_loader, device)

    baseline_group_losses = stage1_ref.compute_subgroup_losses(train_loader, device, cfg.num_subgroups)
    model, stage4_bundle, stage4_checkpoint_paths = train_stage4_proposed(
        model, train_loader, val_loader, device, cfg, usefulness_score, stage1_ref
    )
    test_results = evaluate_all_stage4_checkpoints(
        model, stage4_bundle, test_loader, device, cfg, usefulness_score, stage1_ref
    )
    selected_metric = test_results["_selection"]["selected_checkpoint_metric"]
    if stage4_checkpoint_paths:
        save_stage_checkpoint(cfg, model, "stage4")

    all_results = {
        "stage4_only": True,
        "init_from_run_name": init_cfg.run_name,
        "usefulness_score": usefulness_score,
        "usefulness_meta": usefulness_meta,
        "baseline_val": baseline_val,
        "demo_val": demo_val,
        "baseline_group_losses": baseline_group_losses.detach().cpu(),
        "test_results": test_results,
        "selected_stage4_checkpoint": test_results["_selection"],
        "stage4_checkpoints": {
            metric: {
                "checkpoint_epoch": stage4_bundle[metric]["epoch"],
                "val_metrics": stage4_bundle[metric]["val_metrics"],
                "checkpoint_path": stage4_checkpoint_paths.get(metric),
                "test_min_subgroup_improvement": test_results[metric]["test_min_subgroup_improvement"],
            }
            for metric in STAGE4_CHECKPOINT_METRICS
        },
    }
    run_dir = save_run_results(cfg, all_results)
    export_all_stage4_predictions(
        model, stage4_bundle, cfg, device, run_dir, usefulness_score, selected_metric=selected_metric
    )


def run_pipeline(cfg: Config) -> None:
    if cfg.stage4_only:
        run_pipeline_stage4_only(cfg)
        return
    if cfg.stop_after_stage3:
        run_pipeline_stages_1_to_3(cfg)
        return

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)

    print("========== Proposed Fairness Pipeline ==========")
    print("Device:", device)
    print("Run name:", cfg.run_name)
    print(
        "Architecture: fusion=", cfg.fusion_type,
        "| residual_blocks=", cfg.residual_blocks,
        "| head_layernorm=", cfg.head_layernorm,
    )
    print("Config:", cfg)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg, device)

    model = train_stage1_baseline(model, train_loader, val_loader, device, cfg)
    save_stage_checkpoint(cfg, model, "stage1")
    export_phase0_erm_baseline(cfg, model, device, val_loader, test_loader, seed=cfg.seed)
    stage1_ref = build_stage1_baseline_reference(cfg, model, device)
    baseline_group_losses = stage1_ref.compute_subgroup_losses(train_loader, device, cfg.num_subgroups)

    model = train_stage2_demo(model, train_loader, val_loader, device, cfg, stage1_ref)
    save_stage_checkpoint(cfg, model, "stage2")
    usefulness_score, usefulness_meta, baseline_val, demo_val = stage3_compute_usefulness(
        model, val_loader, device, cfg, stage1_ref
    )
    save_stage123_summary(cfg, usefulness_score, usefulness_meta, baseline_val, demo_val)

    model, stage4_bundle, stage4_checkpoint_paths = train_stage4_proposed(
        model, train_loader, val_loader, device, cfg, usefulness_score, stage1_ref
    )
    test_results = evaluate_all_stage4_checkpoints(
        model, stage4_bundle, test_loader, device, cfg, usefulness_score, stage1_ref
    )
    selected_metric = test_results["_selection"]["selected_checkpoint_metric"]
    if stage4_checkpoint_paths:
        save_stage_checkpoint(cfg, model, "stage4")

    all_results = {
        "usefulness_score": usefulness_score,
        "usefulness_meta": usefulness_meta,
        "baseline_val": baseline_val,
        "demo_val": demo_val,
        "baseline_group_losses": baseline_group_losses.detach().cpu(),
        "test_results": test_results,
        "selected_stage4_checkpoint": test_results["_selection"],
        "stage4_checkpoints": {
            metric: {
                "checkpoint_epoch": stage4_bundle[metric]["epoch"],
                "val_metrics": stage4_bundle[metric]["val_metrics"],
                "checkpoint_path": stage4_checkpoint_paths.get(metric),
                "test_min_subgroup_improvement": test_results[metric]["test_min_subgroup_improvement"],
            }
            for metric in STAGE4_CHECKPOINT_METRICS
        },
    }
    run_dir = save_run_results(cfg, all_results)
    export_all_stage4_predictions(
        model, stage4_bundle, cfg, device, run_dir, usefulness_score, selected_metric=selected_metric
    )


def run_entry(cfg: Config) -> None:
    if cfg.baseline_only:
        run_phase0_baselines(cfg)
        return
    if cfg.finetune_stages:
        if cfg.finetune_sequential_lambdas:
            run_sequential_lambda_finetuning_pipeline(cfg)
        else:
            run_finetuning_pipeline(cfg)
    else:
        run_pipeline(cfg)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="CheXpert staged ContextFair pipeline (ResNet50, multi-label BCE, "
        "5-pathology macro AUC selection; FiLM/scalar fusion, optional residual/LN heads)."
    )
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--checkpoint-dir", type=str, default=Config.checkpoint_dir)
    parser.add_argument("--run-name", type=str, default=Config.run_name)
    parser.add_argument("--backbone", type=str, default=Config.backbone, choices=list(SUPPORTED_BACKBONES))
    parser.add_argument("--demographic-mode", type=str, choices=["sex", "age", "custom"], default=Config.demographic_mode)
    parser.add_argument("--demographic-key", type=str, default="")
    parser.add_argument("--num-subgroups", type=int, default=None)
    parser.add_argument(
        "--train-diseases",
        type=int,
        choices=list(TRAIN_DISEASE_CHOICES),
        default=Config.train_diseases,
        help="Train width: 13 (all CheXpert findings, default) or 5 (competition findings only). "
        "Eval/selection is always the 5 competition pathologies (macro AUC).",
    )
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--feature-dim", type=int, default=Config.feature_dim)
    parser.add_argument("--demo-embed-dim", type=int, default=Config.demo_embed_dim)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument(
        "--fusion-type",
        type=str,
        choices=list(FUSION_TYPES),
        default=Config.fusion_type,
        help="Stage 4 demographic fusion: film | scalar",
    )
    parser.add_argument(
        "--residual-blocks",
        type=int,
        default=Config.residual_blocks,
        help="0 = legacy 1-hidden-layer head; >=1 = stacked residual blocks",
    )
    parser.add_argument(
        "--head-layernorm",
        action="store_true",
        default=Config.head_layernorm,
        help="LayerNorm on legacy input and/or inside residual blocks",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--stage1-epochs", type=int, default=Config.stage1_epochs)
    parser.add_argument("--stage2-epochs", type=int, default=Config.stage2_epochs)
    parser.add_argument("--stage4-epochs", type=int, default=Config.stage4_epochs)
    parser.add_argument("--stage1-lr", type=float, default=Config.stage1_lr)
    parser.add_argument("--stage2-lr", type=float, default=Config.stage2_lr)
    parser.add_argument("--stage4-head-lr", type=float, default=Config.stage4_head_lr)
    parser.add_argument("--stage4-encoder-lr", type=float, default=Config.stage4_encoder_lr)
    parser.add_argument("--stage4-unfreeze-encoder", action="store_true", default=Config.stage4_unfreeze_encoder)
    parser.add_argument("--no-stage4-unfreeze-encoder", action="store_false", dest="stage4_unfreeze_encoder")
    parser.add_argument("--stage4-unfreeze-from-block", type=int, default=Config.stage4_unfreeze_from_block)
    parser.add_argument("--no-stage4-unfreeze-fc", action="store_false", dest="stage4_unfreeze_include_fc")
    parser.add_argument("--use-cosine-schedule-stage1", action="store_true")
    parser.add_argument("--no-cosine-schedule-stage4", action="store_false", dest="use_cosine_schedule_stage4")
    parser.add_argument("--warmup-epochs-stage4", type=int, default=Config.warmup_epochs_stage4)
    parser.add_argument(
        "--stage4-checkpoint-skip-epochs",
        type=int,
        default=Config.stage4_checkpoint_skip_epochs,
        help="Do not consider Stage-4 val checkpoints until after this many epochs (default: 2)",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=Config.min_lr_ratio)
    parser.add_argument("--alpha", type=float, default=Config.alpha)
    parser.add_argument("--lambda-cv", type=float, default=Config.lambda_cv)
    parser.add_argument(
        "--lambda-imp",
        type=float,
        default=Config.lambda_imp,
        help="Weight on Var_g(Δ̃_g): balanced subgroup improvement vs baseline (loss surrogate)",
    )
    parser.add_argument(
        "--lambda-worst",
        type=float,
        default=Config.lambda_worst,
        help="Weight on -min_g(Δ̃_g): raise worst-subgroup improvement floor",
    )
    parser.add_argument(
        "--lambda-fair",
        type=float,
        default=None,
        help="Deprecated alias for --lambda-imp (variance fairness term)",
    )
    parser.add_argument("--lambda-kd", type=float, default=Config.lambda_kd)
    parser.add_argument("--temperature", type=float, default=Config.temperature)
    parser.add_argument("--fairness-use-relu", action="store_true")
    parser.add_argument(
        "--selection-metric-stage1",
        choices=list(SELECTION_METRICS),
        default=Config.selection_metric_stage1,
        help="Stage 1 checkpoint: worst_subgroup | overall_auc | min_improvement (falls back to worst_subgroup)",
    )
    parser.add_argument(
        "--selection-metric-stage2",
        choices=list(SELECTION_METRICS),
        default=Config.selection_metric_stage2,
        help="Stage 2 checkpoint: worst_subgroup | overall_auc | min_improvement (demo vs baseline Δ_g)",
    )
    parser.add_argument(
        "--selection-metric-stage4",
        choices=list(STAGE4_CHECKPOINT_METRICS),
        default=Config.selection_metric_stage4,
        help="Stage 4 checkpoint metric (only min_improvement is saved/evaluated)",
    )
    parser.add_argument("--tie-break", choices=["on", "off"], default="on")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument(
        "--finetune-stages",
        action="store_true",
        help="Grid-search hyperparameters per stage on --seed; writes finetune_summary.json",
    )
    parser.add_argument(
        "--finetune-sequential-lambdas",
        action="store_true",
        help="Modular sequential finetune (one hyperparameter at a time). "
        "Use --finetune-only-stage to limit which stage is trained.",
    )
    parser.add_argument(
        "--finetune-only-stage",
        choices=list(FINETUNE_ONLY_STAGES),
        default=Config.finetune_only_stage,
        help="all=Stages1–3 grid + Stage4 λ sequential; stage4=λ only (loads prior stages from summary)",
    )
    parser.add_argument(
        "--finetune-from-summary",
        type=str,
        default=Config.finetune_from_summary,
        help="Path to finetune_summary.json when finetuning stage2/3/5 without retraining earlier stages",
    )
    parser.add_argument(
        "--no-save-finetune-checkpoints",
        action="store_true",
        help="Do not write finetune checkpoint files to disk",
    )
    parser.add_argument(
        "--erm-hparams-dir",
        type=str,
        default=Config.erm_hparams_dir,
        help=(
            "Dir with finetune_summary_ERM__resnet50.json (Comp_models ERM finetune "
            "outputs). When present, Stage-1 lr/weight_decay/batch_size/epochs grid "
            "search is skipped and those hparams are reused."
        ),
    )
    parser.add_argument(
        "--skip-erm-hparams",
        action="store_true",
        help="Ignore --erm-hparams-dir; always run the Stage-1 hparam grid search",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "Phase 0: train/export only the local image-only ERM baseline (Stage 1) "
            "for --seed, writing checkpoints/ERM + results/ERM fairness-style artifacts; "
            "skip Stages 2-4."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "With --baseline-only: reuse an existing checkpoints/ERM/seed_{seed}/"
            "{backbone}_image_only.pt instead of retraining Stage 1; error if missing. "
            "With normal runs: also skip re-exporting Phase-0 ERM artifacts that already exist."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=Config.seeds_arg,
        help="Shared seeds: 'random' (default; resolves via --seeds-file) or comma-separated ints",
    )
    parser.add_argument("--num-random-seeds", type=int, default=Config.num_random_seeds)
    parser.add_argument(
        "--seeds-file",
        type=str,
        default=Config.seeds_file,
        help="Path to experiment_seeds.json (default: {output_dir}/experiment_seeds.json)",
    )
    parser.add_argument(
        "--stop-after-stage3",
        action="store_true",
        help="Train Stages 1–3 only; save stage1/stage2/stage1_ref + stage123_summary.json (skip Stage 4)",
    )
    parser.add_argument(
        "--stage4-only",
        action="store_true",
        help="Train Stage 4 only; load Stages 1–3 from --init-from-run-name",
    )
    parser.add_argument(
        "--init-from-run-name",
        type=str,
        default=Config.init_from_run_name,
        help="Base run_name (fusion suffix optional) whose Stages 1–3 checkpoints to load for --stage4-only",
    )
    parser.add_argument("--stage1-lr-grid", type=parse_float_grid, default=Config.stage1_lr_grid)
    parser.add_argument("--stage1-weight-decay-grid", type=parse_float_grid, default=Config.stage1_weight_decay_grid)
    parser.add_argument("--stage1-batch-size-grid", type=parse_int_grid, default=Config.stage1_batch_size_grid)
    parser.add_argument("--stage2-demo-embed-dim-grid", type=parse_int_grid, default=Config.stage2_demo_embed_dim_grid)
    parser.add_argument("--stage3-lambda-cv-grid", type=parse_float_grid, default=Config.stage3_lambda_cv_grid)
    parser.add_argument("--stage3-alpha-grid", type=parse_float_grid, default=Config.stage3_alpha_grid)
    parser.add_argument("--stage4-lambda-imp-grid", type=parse_float_grid, default=Config.stage4_lambda_imp_grid)
    parser.add_argument("--stage4-lambda-worst-grid", type=parse_float_grid, default=Config.stage4_lambda_worst_grid)
    parser.add_argument("--stage4-lambda-kd-grid", type=parse_float_grid, default=Config.stage4_lambda_kd_grid)
    parser.add_argument("--stage4-temperature-grid", type=parse_float_grid, default=Config.stage4_temperature_grid)
    parser.add_argument("--stage4-gamma-grid", type=parse_float_grid, default=Config.stage4_gamma_grid)
    parser.add_argument("--stage4-head-lr-grid", type=parse_float_grid, default=Config.stage4_head_lr_grid)
    parser.add_argument("--stage4-encoder-lr-grid", type=parse_float_grid, default=Config.stage4_encoder_lr_grid)
    parser.add_argument("--stage4-epochs-grid", type=parse_int_grid, default=Config.stage4_epochs_grid)
    parser.add_argument(
        "--stage4-unfreeze-from-block-grid",
        type=parse_int_grid,
        default=Config.stage4_unfreeze_from_block_grid,
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.data_root = args.data_root
    cfg.output_dir = args.output_dir
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.run_name = args.run_name
    cfg.backbone = normalize_backbone(args.backbone)
    cfg.demographic_mode = args.demographic_mode
    apply_demographic_mode_defaults(cfg)
    if args.demographic_key:
        cfg.demographic_key = args.demographic_key
    if args.num_subgroups is not None:
        cfg.num_subgroups = args.num_subgroups
    cfg.train_diseases = args.train_diseases
    apply_train_diseases_defaults(cfg)
    cfg.image_size = args.image_size
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.feature_dim = args.feature_dim
    cfg.demo_embed_dim = args.demo_embed_dim
    cfg.gamma = args.gamma
    cfg.fusion_type = normalize_fusion_type(args.fusion_type)
    cfg.residual_blocks = max(0, int(args.residual_blocks))
    cfg.head_layernorm = bool(args.head_layernorm)
    cfg.pretrained = not args.no_pretrained
    cfg.weight_decay = args.weight_decay
    cfg.stage1_epochs = args.stage1_epochs
    cfg.stage2_epochs = args.stage2_epochs
    cfg.stage4_epochs = args.stage4_epochs
    cfg.stage1_lr = args.stage1_lr
    cfg.stage2_lr = args.stage2_lr
    cfg.stage4_head_lr = args.stage4_head_lr
    cfg.stage4_encoder_lr = args.stage4_encoder_lr
    cfg.stage4_unfreeze_encoder = bool(args.stage4_unfreeze_encoder)
    cfg.stage4_unfreeze_from_block = args.stage4_unfreeze_from_block
    cfg.stage4_unfreeze_include_fc = bool(args.stage4_unfreeze_include_fc)
    cfg.use_cosine_schedule_stage1 = bool(args.use_cosine_schedule_stage1)
    cfg.use_cosine_schedule_stage4 = bool(args.use_cosine_schedule_stage4)
    cfg.warmup_epochs_stage4 = args.warmup_epochs_stage4
    cfg.stage4_checkpoint_skip_epochs = args.stage4_checkpoint_skip_epochs
    cfg.min_lr_ratio = args.min_lr_ratio
    cfg.alpha = args.alpha
    cfg.lambda_cv = args.lambda_cv
    cfg.lambda_imp = args.lambda_fair if args.lambda_fair is not None else args.lambda_imp
    cfg.lambda_worst = args.lambda_worst
    cfg.lambda_kd = args.lambda_kd
    cfg.temperature = args.temperature
    cfg.fairness_use_relu = bool(args.fairness_use_relu)
    cfg.selection_metric_stage1 = args.selection_metric_stage1
    cfg.selection_metric_stage2 = args.selection_metric_stage2
    cfg.selection_metric_stage4 = args.selection_metric_stage4
    cfg.use_overall_tie_break = args.tie_break == "on"
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.finetune_stages = bool(args.finetune_stages)
    cfg.finetune_sequential_lambdas = bool(args.finetune_sequential_lambdas)
    cfg.finetune_only_stage = str(args.finetune_only_stage)
    cfg.finetune_from_summary = str(args.finetune_from_summary or "")
    cfg.save_finetune_checkpoints = not args.no_save_finetune_checkpoints and cfg.finetune_only_stage != "stage4"
    cfg.erm_hparams_dir = str(args.erm_hparams_dir or "")
    cfg.skip_erm_hparams = bool(args.skip_erm_hparams)
    cfg.baseline_only = bool(args.baseline_only)
    cfg.skip_baseline = bool(args.skip_baseline)
    cfg.seeds_arg = str(args.seeds)
    cfg.num_random_seeds = int(args.num_random_seeds)
    cfg.seeds_file = str(args.seeds_file or "")
    cfg.stop_after_stage3 = bool(args.stop_after_stage3)
    cfg.stage4_only = bool(args.stage4_only)
    cfg.init_from_run_name = str(args.init_from_run_name or "")
    if cfg.stage4_only and cfg.stop_after_stage3:
        raise ValueError("Use only one of --stop-after-stage3 or --stage4-only.")
    if cfg.stage4_only and not cfg.init_from_run_name:
        raise ValueError("--stage4-only requires --init-from-run-name.")
    cfg.stage1_lr_grid = args.stage1_lr_grid
    cfg.stage1_weight_decay_grid = args.stage1_weight_decay_grid
    cfg.stage1_batch_size_grid = args.stage1_batch_size_grid
    cfg.stage2_demo_embed_dim_grid = args.stage2_demo_embed_dim_grid
    cfg.stage3_lambda_cv_grid = args.stage3_lambda_cv_grid
    cfg.stage3_alpha_grid = args.stage3_alpha_grid
    cfg.stage4_lambda_imp_grid = args.stage4_lambda_imp_grid
    cfg.stage4_lambda_worst_grid = args.stage4_lambda_worst_grid
    cfg.stage4_lambda_kd_grid = args.stage4_lambda_kd_grid
    cfg.stage4_temperature_grid = args.stage4_temperature_grid
    cfg.stage4_gamma_grid = args.stage4_gamma_grid
    cfg.stage4_head_lr_grid = args.stage4_head_lr_grid
    cfg.stage4_encoder_lr_grid = args.stage4_encoder_lr_grid
    cfg.stage4_epochs_grid = args.stage4_epochs_grid
    cfg.stage4_unfreeze_from_block_grid = args.stage4_unfreeze_from_block_grid
    apply_effective_run_name(cfg)
    return cfg


if __name__ == "__main__":
    run_entry(parse_args())
