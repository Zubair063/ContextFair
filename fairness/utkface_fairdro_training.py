from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utkface_fairness_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SELECTION_METRIC,
    NUM_CLASSES,
    SUPPORTED_BACKBONES,
    add_common_arguments,
    add_finetune_arguments,
    apply_demographic_config,
    apply_hparams_to_cfg,
    autocast_ctx,
    build_backbone,
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
    persist_seed_run,
    resolve_seeds,
    resolve_selection_attr,
    run_group_name,
    save_json,
    set_seed,
)


@dataclass
class Config:
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = "utkface_fairdro_outputs"
    erm_hparams_dir: str = ""
    backbone: str = "vgg16"
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224
    seeds: tuple[int, ...] = ()
    demographic_subgroup: str | None = None
    demographic_attrs: tuple[str, ...] = ("age",)
    selection_metric: str = DEFAULT_SELECTION_METRIC
    selection_attr: str = "age"
    baseline_selection_attr: str = "race"
    rho: float = 0.3
    optim_q: str = "smt_ibr"
    q_decay: str = "linear"
    use_01loss: bool = True
    pretrained: bool = True
    use_amp: bool = True
    lr_grid: tuple[float, ...] = ()
    weight_decay_grid: tuple[float, ...] = ()
    batch_size_grid: tuple[int, ...] = ()
    rho_grid: tuple[float, ...] = ()
    finetune_summary: str = ""
    skip_finetune: bool = False
    finetune_only: bool = False
    skip_baseline: bool = False
    baseline_only: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Model: Backbone + Linear Classifier
# ─────────────────────────────────────────────────────────────────────────────

class FairDROModel(nn.Module):
    """Standard backbone + linear classification head.

    FairDRO does not modify the model architecture -- the fairness mechanism
    is entirely in the training loss re-weighting.
    """

    def __init__(self, backbone_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.encoder, self.feat_dim = build_backbone(backbone_name, pretrained)
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


# ─────────────────────────────────────────────────────────────────────────────
# FairDRO q-update (Proposition 2 from the paper)
# ─────────────────────────────────────────────────────────────────────────────

def update_mw_margin(losses: torch.Tensor, rho: float) -> torch.Tensor:
    """Closed-form IBR solution for the chi-squared DRO inner maximizer."""
    n_groups = len(losses)
    mean = losses.mean()
    denom = (losses - mean).norm(2)
    if denom < 1e-12:
        return torch.zeros_like(losses) + 1.0 / n_groups
    q = 1.0 / n_groups + np.sqrt(2.0 * rho / n_groups) * (1.0 / denom) * (losses - mean)
    return q


def q_update_ibr(
    q_dict: dict[int, torch.Tensor],
    subgroup_loss: torch.Tensor,
    n_classes: int,
    rho: float,
) -> dict[int, torch.Tensor]:
    """Standard IBR: replace q with the optimal solution."""
    opt_q: dict[int, torch.Tensor] = {}
    for y in range(n_classes):
        label_group_loss = subgroup_loss[:, y]
        opt_q[y] = update_mw_margin(label_group_loss, rho)
    return opt_q


def q_update_smt_ibr(
    q_dict: dict[int, torch.Tensor],
    subgroup_loss: torch.Tensor,
    n_classes: int,
    rho: float,
    epoch: int,
    total_epochs: int,
    q_decay: str = "linear",
) -> dict[int, torch.Tensor]:
    """Smoothed IBR with decaying step size (Eq. 10 in the paper)."""
    if q_decay == "cos":
        step = 0.5 * (1.0 + np.cos(np.pi * epoch / max(total_epochs, 1)))
    else:
        step = 1.0 - epoch / max(total_epochs, 1)

    for y in range(n_classes):
        label_group_loss = subgroup_loss[:, y]
        opt_q = update_mw_margin(label_group_loss, rho)
        q_dict[y] = q_dict[y] + step * (opt_q - q_dict[y])
    return q_dict


# ─────────────────────────────────────────────────────────────────────────────
# Compute per-subgroup losses and accuracies on the full dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_subgroup_stats(
    model: FairDROModel,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    n_groups: int,
    demo_attrs: tuple[str, ...],
    use_amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (subgroup_loss, subgroup_acc) both of shape (n_groups, n_classes)."""
    n_subgroups = n_classes * n_groups
    group_count = torch.zeros(n_subgroups, device=device)
    group_loss_total = torch.zeros(n_subgroups, device=device)
    group_acc_total = torch.zeros(n_subgroups, device=device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    for batch in loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        groups = build_group_ids(batch, demo_attrs, device)

        with autocast_ctx(use_amp, device):
            outputs = model(inputs)
        outputs = outputs.float()
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float()

        subgroups = groups * n_classes + labels
        group_map = (
            subgroups.unsqueeze(0) == torch.arange(n_subgroups, device=device).unsqueeze(1)
        ).float()
        group_count += group_map.sum(1)
        group_loss_total += group_map @ loss
        group_acc_total += group_map @ acc

    group_denom = group_count + (group_count == 0).float()
    subgroup_loss = (group_loss_total / group_denom).reshape(n_groups, n_classes)
    subgroup_acc = (group_acc_total / group_denom).reshape(n_groups, n_classes)

    model.train()
    return subgroup_loss, subgroup_acc


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: FairDROModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    baseline_report: dict[str, float] | None,
) -> tuple[FairDROModel, dict[str, float]]:
    n_classes = NUM_CLASSES
    n_groups = compute_n_groups(cfg.demographic_attrs)
    n_subgroups = n_classes * n_groups

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(reduction="none")
    scaler = make_grad_scaler(cfg.use_amp, device)

    q_dict: dict[int, torch.Tensor] = {}
    for y in range(n_classes):
        q_dict[y] = torch.ones(n_groups, device=device) / n_groups

    eval_kw: dict[str, Any] = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        # UTKFace `label` is added by collate_fn (not Dataset.__getitem__).
        collate_fn=train_loader.collate_fn,
    )
    if cfg.num_workers > 0:
        eval_kw["persistent_workers"] = True
    eval_train_loader = DataLoader(train_loader.dataset, **eval_kw)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_report: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            groups = build_group_ids(batch, cfg.demographic_attrs, device)

            with autocast_ctx(cfg.use_amp, device):
                outputs = model(inputs)
            outputs = outputs.float()
            loss_per_sample = criterion(outputs, labels)

            subgroups = groups * n_classes + labels
            group_map = (
                subgroups.unsqueeze(0) == torch.arange(n_subgroups, device=device).unsqueeze(1)
            ).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count == 0).float()
            group_loss = (group_map @ loss_per_sample) / group_denom
            group_loss = group_loss.reshape(n_groups, n_classes)

            robust_loss = torch.tensor(0.0, device=device)
            for y in range(n_classes):
                robust_loss = robust_loss + group_loss[:, y] @ q_dict[y]
            robust_loss = robust_loss / n_classes

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(robust_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)
            batch_acc = (preds == labels).float().mean().item()
            running_loss += robust_loss.item()
            running_acc += batch_acc
            n_batches += 1
            pbar.set_postfix({
                "loss": f"{robust_loss.item():.4f}",
                "acc": f"{batch_acc:.3f}",
            })

        subgroup_loss, subgroup_acc = compute_subgroup_stats(
            model, eval_train_loader, device, n_classes, n_groups,
            cfg.demographic_attrs, cfg.use_amp,
        )

        loss_for_q = (1.0 - subgroup_acc) if cfg.use_01loss else subgroup_loss

        if cfg.optim_q == "ibr":
            q_dict = q_update_ibr(q_dict, loss_for_q, n_classes, cfg.rho)
        else:
            q_dict = q_update_smt_ibr(
                q_dict, loss_for_q, n_classes, cfg.rho,
                epoch, cfg.epochs, cfg.q_decay,
            )

        scheduler.step()

        val_report = evaluate_subgroup_report(
            model, val_loader, device, cfg.selection_attr, use_amp=cfg.use_amp
        )
        score = compute_selection_score(
            val_report,
            selection_metric=cfg.selection_metric,
            selection_attr=cfg.selection_attr,
            baseline_report=baseline_report,
        )
        val_report["selection_score"] = float(score) if not math.isnan(score) else float("nan")

        avg_loss = running_loss / max(n_batches, 1)
        avg_acc = running_acc / max(n_batches, 1)
        q_str = "  ".join(
            f"q[y={y}]=[{', '.join(f'{q_dict[y][g]:.3f}' for g in range(n_groups))}]"
            for y in range(n_classes)
        )
        print(
            f"Epoch {epoch + 1:03d}: train_loss={avg_loss:.4f}, "
            f"train_acc={avg_acc:.3f}, overall_acc={val_report['overall_acc']:.4f}, "
            f"{cfg.selection_metric}={score:.4f}  {q_str}"
        )

        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_report = dict(val_report)

    model.load_state_dict(best_state)
    if not best_report:
        best_report = evaluate_subgroup_report(
            model, val_loader, device, cfg.selection_attr, use_amp=cfg.use_amp
        )
        best_report["selection_score"] = compute_selection_score(
            best_report,
            selection_metric=cfg.selection_metric,
            selection_attr=cfg.selection_attr,
            baseline_report=baseline_report,
        )
    return model, best_report


# ─────────────────────────────────────────────────────────────────────────────
# CLI & Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FairDRO (Fusion-style finetune, ACC-based) for UTKFace."
    )
    add_common_arguments(
        parser,
        default_output_dir="utkface_fairdro_outputs",
        default_backbone="vit_b_16",
        default_epochs=20,
        default_batch_size=128,
    )
    add_finetune_arguments(parser)
    parser.add_argument(
        "--rho", type=float, default=Config.rho,
        help="Radius of chi-squared divergence ball (controls fairness strength)",
    )
    parser.add_argument(
        "--optim-q", type=str, default=Config.optim_q, choices=["smt_ibr", "ibr"],
        help="Q-update method: smt_ibr (smoothed) or ibr",
    )
    parser.add_argument(
        "--q-decay", type=str, default=Config.q_decay, choices=["linear", "cos"],
        help="Decay schedule for smoothed IBR step size",
    )
    parser.add_argument(
        "--use-01loss", action="store_true", default=True,
        help="Use 0-1 loss for q update (default: True)",
    )
    parser.add_argument(
        "--use-ce-for-q", dest="use_01loss", action="store_false",
        help="Use CE loss instead of 0-1 loss for q update",
    )
    parser.add_argument("--rho-grid", type=str, default="0.1,0.3,0.5")
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
    seeds_file = Path(args.seeds_file) if str(args.seeds_file).strip() else (out_dir / "experiment_seeds.json")
    cfg.seeds = resolve_seeds(
        seeds_arg=args.seeds,
        num_random_seeds=args.num_random_seeds,
        seeds_file=seeds_file if seeds_file.is_file() else None,
        save_seeds_file=seeds_file,
    )
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers
    cfg.image_size = args.image_size
    cfg.rho = args.rho
    cfg.optim_q = args.optim_q
    cfg.q_decay = args.q_decay
    cfg.use_01loss = args.use_01loss
    cfg.pretrained = not args.no_pretrained

    cfg.use_amp = True
    if args.no_use_amp:
        cfg.use_amp = False
    elif args.use_amp:
        cfg.use_amp = True

    cfg.lr_grid = parse_float_grid(args.lr_grid)
    cfg.weight_decay_grid = parse_float_grid(args.weight_decay_grid)
    cfg.batch_size_grid = tuple(int(x) for x in parse_float_grid(args.batch_size_grid))
    cfg.rho_grid = parse_float_grid(args.rho_grid)
    cfg.finetune_summary = args.finetune_summary
    cfg.skip_finetune = bool(args.skip_finetune)
    cfg.finetune_only = bool(args.finetune_only)
    cfg.skip_baseline = bool(args.skip_baseline)
    cfg.baseline_only = bool(args.baseline_only)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {cfg.backbone}")
    print(f"Seeds: {cfg.seeds}")
    print(f"ERM hparams dir: {cfg.erm_hparams_dir or '(REQUIRED)'}")
    print(f"Selection metric: {cfg.selection_metric}")
    print(f"Baseline selection_attr={cfg.baseline_selection_attr}")
    print(
        f"FairDRO: optim_q={cfg.optim_q}, q_decay={cfg.q_decay}, "
        f"use_01loss={cfg.use_01loss}, use_amp={cfg.use_amp}"
    )

    if cfg.baseline_only or needs_local_baseline(cfg.selection_metric):
        ensure_baseline_erm(cfg, cfg.backbone, device)
    if cfg.baseline_only:
        print("baseline_only=1; skipping FairDRO training.")
        return

    run_group = run_group_name("FairDRO", cfg.demographic_attrs, cfg.demographic_subgroup)
    ckpt_root, results_root = get_output_roots(
        cfg.output_dir, "FairDRO", cfg.demographic_attrs, cfg.demographic_subgroup
    )
    print(f"Demographic attrs: {cfg.demographic_attrs} | selection_attr={cfg.selection_attr}")
    print(f"Checkpoint metric: {cfg.selection_metric}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader_cache: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {
        int(cfg.batch_size): (train_loader, val_loader, test_loader),
    }

    def _loaders_for(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        bs = int(batch_size)
        if bs not in loader_cache:
            tmp = copy.deepcopy(cfg)
            tmp.batch_size = bs
            loader_cache[bs] = build_dataloaders(tmp)
            print(f"Rebuilt dataloaders for batch_size={bs}")
        return loader_cache[bs]

    mid_wd = mid_of(cfg.weight_decay_grid, cfg.weight_decay)
    mid_bs = mid_of(cfg.batch_size_grid, cfg.batch_size)
    mid_rho = mid_of(cfg.rho_grid, cfg.rho)
    initial_hparams: dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": float(mid_wd),
        "batch_size": int(mid_bs),
        "epochs": cfg.epochs,
        "rho": float(mid_rho),
    }

    def trial_fn(hparams: dict[str, Any], stage: str) -> dict[str, Any]:
        trial_cfg = apply_hparams_to_cfg(cfg, hparams)
        set_seed(int(cfg.seeds[0]))
        tr, va, _te = _loaders_for(int(trial_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(
            trial_cfg, trial_cfg.backbone, int(cfg.seeds[0]), va, device
        )
        print(f"\n--- Finetune [{stage}] hparams={hparams} ---")
        model = FairDROModel(trial_cfg.backbone, pretrained=trial_cfg.pretrained).to(device)
        model, report = train_model(model, tr, va, device, trial_cfg, baseline_report)
        score = float(report.get("selection_score", float("nan")))
        return {"stage": stage, "hparams": dict(hparams), "metrics": report, "selection_score": score}

    stages = [
        ("lr", cfg.lr_grid),
        ("weight_decay", cfg.weight_decay_grid),
        ("batch_size", cfg.batch_size_grid),
        ("rho", cfg.rho_grid),
    ]
    grids_meta = {
        "lr": list(cfg.lr_grid),
        "weight_decay": list(cfg.weight_decay_grid),
        "batch_size": list(cfg.batch_size_grid),
        "rho": list(cfg.rho_grid),
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
    train_loader, val_loader, test_loader = _loaders_for(int(train_cfg.batch_size))
    print(f"Training all seeds with hparams: {summary['hparams']}")

    for seed in cfg.seeds:
        print("=" * 80)
        print(f"FairDRO | seed={seed} | backbone={train_cfg.backbone} | attrs={train_cfg.demographic_attrs}")
        set_seed(int(seed))
        tr, va, te = _loaders_for(int(train_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(
            train_cfg, train_cfg.backbone, int(seed), va, device
        )
        model = FairDROModel(train_cfg.backbone, pretrained=train_cfg.pretrained).to(device)
        model, best_report = train_model(model, tr, va, device, train_cfg, baseline_report)

        attr_suffix = "_".join(train_cfg.demographic_attrs)
        run_name = f"{train_cfg.backbone}_FairDRO_{attr_suffix}"
        rows = collect_test_rows(model, te, device, run_name, int(seed), train_cfg.use_amp)
        csv_path, ckpt_path = persist_seed_run(
            results_root,
            ckpt_root,
            int(seed),
            model.state_dict(),
            f"fairdro_{train_cfg.backbone}.pt",
            rows,
        )
        save_json(
            ckpt_path.parent / f"fairdro_{train_cfg.backbone}_val_metrics.json",
            {
                "seed": int(seed),
                "selection_metric": train_cfg.selection_metric,
                "selection_attr": train_cfg.selection_attr,
                "hparams": summary["hparams"],
                "val_metrics": best_report,
            },
        )
        print(f"Saved checkpoint: {ckpt_path}")
        print(f"Saved predictions: {csv_path}")

    print(f"\nFairDRO complete → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
