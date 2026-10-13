from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utkface_fairness_common import (  # noqa: E402
    DEFAULT_BACKBONE,
    DEFAULT_DATA_ROOT,
    DEFAULT_SELECTION_METRIC,
    NUM_CLASSES,
    SUPPORTED_BACKBONES,
    add_common_arguments,
    add_finetune_arguments,
    apply_demographic_config,
    apply_hparams_to_cfg,
    autocast_ctx,
    build_dataloaders,
    build_group_ids,
    collect_test_rows,
    compute_n_groups,
    compute_selection_score,
    ensure_baseline_erm,
    evaluate_subgroup_report,
    get_output_roots,
    load_or_run_finetune,
    maybe_load_baseline_report,
    make_grad_scaler,
    mid_of,
    needs_local_baseline,
    normalize_selection_metric,
    parse_float_grid,
    resolve_seeds,
    resolve_selection_attr,
    run_group_name,
    save_json,
    save_seed_results,
    set_seed,
    setup_cudnn_benchmark,
)

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@dataclass
class Config:
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = "utkface_fairtune_outputs"
    erm_hparams_dir: str = ""
    backbone: str = DEFAULT_BACKBONE
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    seeds: tuple[int, ...] = ()
    demographic_subgroup: str | None = None
    demographic_attrs: tuple[str, ...] = ("age",)
    selection_metric: str = DEFAULT_SELECTION_METRIC
    selection_attr: str = "age"
    baseline_selection_attr: str = "race"
    pretrained: bool = True
    lr: float = 1e-4
    weight_decay: float = 1e-4
    search_trials: int = 30
    search_epochs: int = 5
    finetune_epochs: int = 20
    lr_search_low: float = 1e-5
    lr_search_high: float = 1e-3
    finetune_lr: float | None = None
    use_amp: bool = False
    cudnn_benchmark: bool = False
    lr_grid: tuple[float, ...] = ()
    weight_decay_grid: tuple[float, ...] = ()
    batch_size_grid: tuple[int, ...] = ()
    finetune_summary: str = ""
    skip_finetune: bool = False
    finetune_only: bool = False
    skip_baseline: bool = False
    baseline_only: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Backbone + maskable groups (PEFT-style block selection, FairTune-style)
# ─────────────────────────────────────────────────────────────────────────────


def build_encoder_and_head(
    name: str, num_classes: int, pretrained: bool,
) -> tuple[nn.Module, nn.Module, int, list[nn.Module]]:
    """Returns (encoder, head, feat_dim, maskable_groups).

    maskable_groups: list of nn.Module subtrees; binary mask selects which to train.
    Stem / early layers stay frozen when mask bit is 0 (FairTune: non-selected blocks frozen).
    """
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
        groups = [m.layer1, m.layer2, m.layer3, m.layer4]
        return m, nn.Linear(feat_dim, num_classes), feat_dim, groups

    if name == "densenet121":
        w = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = densenet121(weights=w)
        feat_dim = m.classifier.in_features
        m.classifier = nn.Identity()
        ch = list(m.features.children())
        n = len(ch)
        cuts = [0, n // 4, n // 2, 3 * n // 4, n]
        groups = [nn.Sequential(*ch[cuts[i] : cuts[i + 1]]) for i in range(4)]
        return m, nn.Linear(feat_dim, num_classes), feat_dim, groups

    if name == "efficientnet_b0":
        w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b0(weights=w)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        ch = list(m.features.children())
        n = len(ch)
        cuts = [0, n // 4, n // 2, 3 * n // 4, n]
        groups = [nn.Sequential(*ch[cuts[i] : cuts[i + 1]]) for i in range(4)]
        return m, nn.Linear(feat_dim, num_classes), feat_dim, groups

    if name == "vgg16":
        w = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        m = vgg16(weights=w)
        feat_dim = m.classifier[6].in_features
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
        ch = list(m.features.children())
        n = len(ch)
        cuts = [0, n // 4, n // 2, 3 * n // 4, n]
        groups = [nn.Sequential(*ch[cuts[i] : cuts[i + 1]]) for i in range(4)]
        return m, nn.Linear(feat_dim, num_classes), feat_dim, groups

    if name == "vit_b_16":
        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = vit_b_16(weights=w)
        feat_dim = m.heads.head.in_features
        m.heads = nn.Identity()
        layers = list(m.encoder.layers)
        groups = layers
        return m, nn.Linear(feat_dim, num_classes), feat_dim, groups

    raise ValueError(f"Unsupported backbone '{name}'. Choose from: {SUPPORTED_BACKBONES}")


class FairTuneModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.encoder, self.head, self.feat_dim, self.mask_groups = build_encoder_and_head(
            backbone, num_classes, pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)


def freeze_all_encoder(model: FairTuneModel) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False


def apply_mask(model: FairTuneModel, mask: Sequence[int]) -> None:
    """mask[i] in {0,1}: train encoder mask_groups[i] if 1. Head always trainable."""
    freeze_all_encoder(model)
    for p in model.head.parameters():
        p.requires_grad = True
    for bit, mod in zip(mask, model.mask_groups):
        if int(bit) == 1:
            for p in mod.parameters():
                p.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: FairTuneModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    desc: str | None = None,
) -> float:
    model.train()
    if scaler is None:
        scaler = make_grad_scaler(use_amp, device)
    loss_sum, n = 0.0, 0
    iterator: Any = loader
    if desc is not None:
        iterator = tqdm(loader, desc=desc, leave=False)
    for batch in iterator:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(use_amp, device):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs
        if desc is not None:
            iterator.set_postfix(loss=f"{loss.item():.4f}")
    return loss_sum / max(n, 1)


@torch.no_grad()
def collect_predictions_by_group(
    model: FairTuneModel,
    loader: DataLoader,
    device: torch.device,
    demo_attrs: tuple[str, ...],
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns labels (N,), probs (N,), group_ids (N,) on CPU."""
    model.eval()
    labels_l: list[np.ndarray] = []
    probs_l: list[np.ndarray] = []
    groups_l: list[np.ndarray] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        g = build_group_ids(batch, demo_attrs, device)
        with autocast_ctx(use_amp, device):
            logits = model(x)
        pr = F.softmax(logits.float(), dim=1)[:, 1]
        labels_l.append(y.cpu().numpy())
        probs_l.append(pr.cpu().numpy())
        groups_l.append(g.cpu().numpy())
    return (
        np.concatenate(labels_l),
        np.concatenate(probs_l),
        np.concatenate(groups_l),
    )


def subgroup_auc_disparity(
    labels: np.ndarray, probs: np.ndarray, group_ids: np.ndarray, n_groups: int,
) -> tuple[float, float]:
    """Returns (overall_auc, auc_gap=max_auc-min_auc over groups with valid AUC).

    This is FairTune's original Stage-1 (mask search) objective and is left
    unchanged: the outer Fusion trial/selection score uses ACC-based
    overall_acc/min_improvement/worst_subgroup instead (see
    ``train_finetune_epochs``), but the mask search itself still minimizes
    this AUC gap.
    """
    try:
        overall = float(roc_auc_score(labels, probs))
    except Exception:
        overall = float("nan")
    aucs: list[float] = []
    for g in range(n_groups):
        m = group_ids == g
        if m.sum() < 5:
            continue
        yg = labels[m]
        pg = probs[m]
        if len(np.unique(yg)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yg, pg)))
        except Exception:
            continue
    if len(aucs) < 2:
        gap = 0.0
    else:
        gap = float(max(aucs) - min(aucs))
    return overall, gap


def train_finetune_epochs(
    model: FairTuneModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    baseline_report: dict[str, float] | None,
    selection_attr: str,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    use_amp: bool = False,
) -> tuple[FairTuneModel, dict[str, float]]:
    """Stage 2 fine-tune. Checkpointing uses ACC-based selection (overall_acc /
    min_improvement vs the Fusion local same-seed image-only ERM baseline /
    worst_subgroup), replacing the original best-overall-AUC checkpointing.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scaler = make_grad_scaler(use_amp, device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_report: dict[str, float] = {}

    for ep in range(epochs):
        train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            use_amp=use_amp,
            scaler=scaler,
            desc=f"FairTune fine-tune {ep + 1}/{epochs}",
        )
        val_report = evaluate_subgroup_report(
            model, val_loader, device, selection_attr, use_amp=use_amp,
        )
        score = compute_selection_score(
            val_report,
            selection_metric=selection_metric,
            selection_attr=selection_attr,
            baseline_report=baseline_report,
        )
        val_report["selection_score"] = float(score) if not math.isnan(score) else float("nan")
        print(
            f"FairTune Epoch {ep + 1:03d}: overall_acc={val_report['overall_acc']:.4f}, "
            f"{selection_metric}={score:.4f}"
        )
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_report = dict(val_report)

    model.load_state_dict(best_state)
    if not best_report:
        best_report = evaluate_subgroup_report(
            model, val_loader, device, selection_attr, use_amp=use_amp,
        )
        best_report["selection_score"] = compute_selection_score(
            best_report,
            selection_metric=selection_metric,
            selection_attr=selection_attr,
            baseline_report=baseline_report,
        )
    return model, best_report


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: mask + LR search (Optuna TPE, or random fallback) — unchanged algorithm
# ─────────────────────────────────────────────────────────────────────────────


def run_mask_search(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_mask_bits: int,
    seed: int,
    save_dir: Path,
) -> tuple[np.ndarray, float]:
    """Returns (best_mask int8 array, best_lr). Original FairTune objective
    (minimize val subgroup AUC gap); untouched by the outer Fusion protocol.
    """
    n_groups_metric = compute_n_groups(cfg.demographic_attrs)

    def evaluate_trial(mask: list[int], lr: float) -> float:
        set_seed(seed)
        model = FairTuneModel(cfg.backbone, num_classes=NUM_CLASSES, pretrained=cfg.pretrained).to(device)
        apply_mask(model, mask)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return 1e6
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        scaler = make_grad_scaler(cfg.use_amp, device)
        for _ in range(cfg.search_epochs):
            train_one_epoch(
                model, train_loader, device, optimizer,
                use_amp=cfg.use_amp, scaler=scaler,
            )
        labels, probs, gids = collect_predictions_by_group(
            model, val_loader, device, cfg.demographic_attrs, use_amp=cfg.use_amp,
        )
        _, gap = subgroup_auc_disparity(labels, probs, gids, n_groups_metric)
        return gap

    best_mask: list[int] | None = None
    best_lr = 1e-4
    best_score = float("inf")

    if HAS_OPTUNA:

        def objective(trial: optuna.Trial) -> float:
            mask = [trial.suggest_int(f"m{i}", 0, 1) for i in range(n_mask_bits)]
            lr = trial.suggest_float("lr", cfg.lr_search_low, cfg.lr_search_high, log=True)
            return evaluate_trial(mask, lr)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=cfg.search_trials, show_progress_bar=False)
        best_mask = [study.best_trial.params[f"m{i}"] for i in range(n_mask_bits)]
        best_lr = float(study.best_trial.params["lr"])
        best_score = float(study.best_value)
        print(f"[FairTune search] best auc_gap={best_score:.4f}, lr={best_lr:.2e}, mask={best_mask}")
    else:
        rng = random.Random(seed)
        print("Optuna not installed; using random mask search.")
        for t in range(cfg.search_trials):
            mask = [rng.randint(0, 1) for _ in range(n_mask_bits)]
            log_lo = np.log(cfg.lr_search_low)
            log_hi = np.log(cfg.lr_search_high)
            lr = float(np.exp(rng.uniform(log_lo, log_hi)))
            score = evaluate_trial(mask, lr)
            if score < best_score:
                best_score = score
                best_mask = mask
                best_lr = lr
            if (t + 1) % 5 == 0:
                print(f"  trial {t + 1}/{cfg.search_trials} best_auc_gap={best_score:.4f}")
        assert best_mask is not None
        print(f"[FairTune search] best auc_gap={best_score:.4f}, lr={best_lr:.2e}, mask={best_mask}")

    mask_arr = np.array(best_mask, dtype=np.int8)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "best_mask.npy", mask_arr)
    save_json(
        save_dir / "search_best.json",
        {"best_lr": best_lr, "best_val_auc_gap": best_score, "mask": best_mask},
    )
    return mask_arr, best_lr


# ─────────────────────────────────────────────────────────────────────────────
# Full two-stage train (Stage 1 mask search → Stage 2 fine-tune), reused for
# both finetune trials and final multi-seed training.
# ─────────────────────────────────────────────────────────────────────────────


def train_fairtune(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    baseline_report: dict[str, float] | None,
    n_mask_bits: int,
    seed: int,
    mask_save_dir: Path,
) -> tuple[FairTuneModel, np.ndarray, float, dict[str, float]]:
    best_mask, search_lr = run_mask_search(
        cfg, train_loader, val_loader, device, n_mask_bits, seed, mask_save_dir,
    )
    ft_lr = cfg.finetune_lr if cfg.finetune_lr is not None else search_lr

    set_seed(seed)
    model = FairTuneModel(cfg.backbone, num_classes=NUM_CLASSES, pretrained=cfg.pretrained).to(device)
    apply_mask(model, best_mask.tolist())
    tr_params, tot_params = trainable_parameter_count(model)
    print(
        f"Fine-tune: trainable params {tr_params:,} / {tot_params:,} "
        f"({100 * tr_params / tot_params:.2f}%), lr={ft_lr}"
    )
    model, best_report = train_finetune_epochs(
        model, train_loader, val_loader, device,
        cfg.finetune_epochs, float(ft_lr), cfg.weight_decay,
        baseline_report, cfg.selection_attr,
        selection_metric=cfg.selection_metric, use_amp=cfg.use_amp,
    )
    return model, best_mask, float(ft_lr), best_report


# ─────────────────────────────────────────────────────────────────────────────
# CLI & Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FairTune (Fusion-style finetune, ACC-based) for UTKFace.",
    )
    add_common_arguments(
        parser,
        default_output_dir=str(_SCRIPT_DIR / "utkface_fairtune_outputs"),
        default_backbone=DEFAULT_BACKBONE,
        default_epochs=20,
        default_batch_size=32,
    )
    add_finetune_arguments(parser)
    parser.add_argument("--search-trials", type=int, default=Config.search_trials)
    parser.add_argument("--search-epochs", type=int, default=Config.search_epochs)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--lr-search-low", type=float, default=Config.lr_search_low)
    parser.add_argument("--lr-search-high", type=float, default=Config.lr_search_high)
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.data_root = args.data_root
    cfg.output_dir = args.output_dir
    cfg.erm_hparams_dir = args.erm_hparams_dir
    cfg.backbone = args.backbone.lower()
    if cfg.backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone '{cfg.backbone}'. Choose from: {SUPPORTED_BACKBONES}")
    apply_demographic_config(cfg, args.demographic_subgroup, args.demographic_attrs)
    cfg.selection_metric = normalize_selection_metric(args.selection_metric)
    cfg.selection_attr = resolve_selection_attr(args.selection_attr, cfg.demographic_attrs)
    cfg.baseline_selection_attr = resolve_selection_attr(
        args.baseline_selection_attr, ("race",)
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_file = Path(args.seeds_file) if str(args.seeds_file).strip() else (out_dir / "experiment_seeds.json")
    cfg.seeds = resolve_seeds(
        seeds_arg=args.seeds,
        num_random_seeds=args.num_random_seeds,
        seeds_file=seeds_file if seeds_file.is_file() else None,
        save_seeds_file=seeds_file,
    )

    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.image_size = args.image_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.search_trials = args.search_trials
    cfg.search_epochs = args.search_epochs
    cfg.finetune_epochs = args.finetune_epochs if args.finetune_epochs is not None else args.epochs
    cfg.lr_search_low = args.lr_search_low
    cfg.lr_search_high = args.lr_search_high
    cfg.pretrained = not args.no_pretrained
    cfg.cudnn_benchmark = bool(args.cudnn_benchmark)

    cfg.use_amp = False
    if args.no_use_amp:
        cfg.use_amp = False
    elif args.use_amp:
        cfg.use_amp = True

    cfg.lr_grid = parse_float_grid(args.lr_grid)
    cfg.weight_decay_grid = parse_float_grid(args.weight_decay_grid)
    cfg.batch_size_grid = tuple(int(x) for x in parse_float_grid(args.batch_size_grid))
    cfg.finetune_summary = args.finetune_summary
    cfg.skip_finetune = bool(args.skip_finetune)
    cfg.finetune_only = bool(args.finetune_only)
    cfg.skip_baseline = bool(args.skip_baseline)
    cfg.baseline_only = bool(args.baseline_only)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_cudnn_benchmark(cfg.cudnn_benchmark, device)

    print(f"Using device: {device}")
    print(f"Backbone: {cfg.backbone}, pretrained={cfg.pretrained}")
    print(f"Seeds: {cfg.seeds}")
    print(f"ERM hparams dir: {cfg.erm_hparams_dir or '(REQUIRED)'}")
    print(f"Selection metric: {cfg.selection_metric}")
    print(f"Baseline selection_attr={cfg.baseline_selection_attr}")
    print(f"Optuna: {HAS_OPTUNA}, AMP: {cfg.use_amp}")
    print(f"Search: trials={cfg.search_trials} epochs={cfg.search_epochs} (fixed, not gridded)")
    if not HAS_OPTUNA:
        print("Install optuna for TPE-based mask search: pip install optuna")

    if cfg.baseline_only or needs_local_baseline(cfg.selection_metric):
        ensure_baseline_erm(cfg, cfg.backbone, device)
    if cfg.baseline_only:
        print("baseline_only=1; skipping FairTune training.")
        return

    run_group = run_group_name("FairTune", cfg.demographic_attrs, cfg.demographic_subgroup)
    ckpt_root, results_root = get_output_roots(
        cfg.output_dir, "FairTune", cfg.demographic_attrs, cfg.demographic_subgroup,
    )
    print(f"Demographic attrs: {cfg.demographic_attrs} | selection_attr={cfg.selection_attr}")
    print(
        f"Checkpoint metric: {cfg.selection_metric} (fine-tune stage); "
        "mask search stage keeps its original val subgroup AUC-gap objective"
    )

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader_cache: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {
        int(cfg.batch_size): (train_loader, val_loader, test_loader),
    }

    def _loaders_for(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        bs = int(batch_size)
        if bs not in loader_cache:
            tmp_cfg = copy.deepcopy(cfg)
            tmp_cfg.batch_size = bs
            loader_cache[bs] = build_dataloaders(tmp_cfg)
            print(f"Rebuilt dataloaders for batch_size={bs}")
        return loader_cache[bs]

    # Infer mask length from a throwaway model (backbone-dependent only).
    tmp_model = FairTuneModel(cfg.backbone, NUM_CLASSES, cfg.pretrained)
    n_bits = len(tmp_model.mask_groups)
    del tmp_model

    mid_wd = mid_of(cfg.weight_decay_grid, cfg.weight_decay)
    mid_bs = mid_of(cfg.batch_size_grid, cfg.batch_size)
    initial_hparams: dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": float(mid_wd),
        "batch_size": int(mid_bs),
    }

    def trial_fn(hparams: dict[str, Any], stage: str) -> dict[str, Any]:
        trial_cfg = apply_hparams_to_cfg(cfg, hparams)
        trial_cfg.finetune_lr = float(trial_cfg.lr)
        set_seed(int(cfg.seeds[0]))
        tr, va, _te = _loaders_for(int(trial_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(
            trial_cfg, trial_cfg.backbone, int(cfg.seeds[0]), va, device
        )
        print(f"\n--- Finetune [{stage}] hparams={hparams} ---")
        trial_mask_dir = Path(trial_cfg.output_dir) / "finetune" / "mask_search_trial"
        _model, _mask, _ft_lr, report = train_fairtune(
            trial_cfg, tr, va, device, baseline_report, n_bits, int(cfg.seeds[0]), trial_mask_dir,
        )
        score = float(report.get("selection_score", float("nan")))
        return {"stage": stage, "hparams": dict(hparams), "metrics": report, "selection_score": score}

    stages = [
        ("lr", cfg.lr_grid),
        ("weight_decay", cfg.weight_decay_grid),
        ("batch_size", cfg.batch_size_grid),
    ]
    grids_meta = {
        "lr": list(cfg.lr_grid),
        "weight_decay": list(cfg.weight_decay_grid),
        "batch_size": list(cfg.batch_size_grid),
    }
    summary = load_or_run_finetune(
        cfg=cfg,
        run_group=run_group,
        model_name=cfg.backbone,
        stages=stages,
        initial_hparams=initial_hparams,
        trial_fn=trial_fn,
        grids_meta=grids_meta,
    )
    if cfg.finetune_only:
        print("finetune_only=1; skipping multi-seed training.")
        return

    train_cfg = apply_hparams_to_cfg(cfg, summary["hparams"])
    train_cfg.finetune_lr = float(train_cfg.lr)
    train_loader, val_loader, test_loader = _loaders_for(int(train_cfg.batch_size))
    print(f"Training all seeds with hparams: {summary['hparams']}")

    for seed in cfg.seeds:
        print("=" * 80)
        print(f"FairTune | seed={seed} | backbone={train_cfg.backbone} | attrs={train_cfg.demographic_attrs}")
        print("=" * 80)
        set_seed(int(seed))
        tr, va, te = _loaders_for(int(train_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(
            train_cfg, train_cfg.backbone, int(seed), va, device
        )

        seed_ckpt_dir = ckpt_root / f"seed_{seed}"
        seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model, best_mask, ft_lr, best_report = train_fairtune(
            train_cfg, tr, va, device, baseline_report, n_bits, int(seed), seed_ckpt_dir,
        )
        torch.save(
            {"state_dict": model.state_dict(), "mask": best_mask, "lr": ft_lr},
            seed_ckpt_dir / f"fairtune_{train_cfg.backbone}.pt",
        )

        attr_suffix = "_".join(train_cfg.demographic_attrs)
        run_name = f"{train_cfg.backbone}_FairTune_{attr_suffix}"
        rows = collect_test_rows(
            model, te, device, run_name, int(seed), use_amp=train_cfg.use_amp,
        )
        csv_path = save_seed_results(results_root, seed, rows)
        save_json(
            seed_ckpt_dir / f"fairtune_{train_cfg.backbone}_val_metrics.json",
            {
                "seed": int(seed),
                "selection_metric": train_cfg.selection_metric,
                "selection_attr": train_cfg.selection_attr,
                "hparams": summary["hparams"],
                "val_metrics": best_report,
            },
        )
        print(f"Saved checkpoint: {seed_ckpt_dir / f'fairtune_{train_cfg.backbone}.pt'}")
        print(f"Saved predictions: {csv_path}")

    print(f"\nFairTune complete → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
