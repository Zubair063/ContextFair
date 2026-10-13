from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from chexpert_fairness_common import (
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
    build_joint_group_ids,
    collect_test_rows,
    compute_n_groups,
    compute_selection_score,
    ensure_baseline_erm,
    evaluate_subgroup_report,
    flatten_multilabel,
    get_output_roots,
    load_or_run_finetune,
    make_grad_scaler,
    maybe_load_baseline_report,
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
    output_dir: str = "chexpert_groupdro_outputs"
    erm_hparams_dir: str = ""
    backbone: str = "resnet50"
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224
    seeds: tuple[int, ...] = ()
    demographic_subgroup: str | None = None
    demographic_attrs: tuple[str, ...] = ("sex",)
    selection_metric: str = DEFAULT_SELECTION_METRIC
    selection_attr: str = "sex"
    baseline_selection_attr: str = "sex"
    robust_step_size: float = 0.01
    gamma: float = 0.1
    generalization_adjustment: float = 0.0
    pretrained: bool = True
    use_amp: bool = True
    lr_grid: tuple[float, ...] = ()
    weight_decay_grid: tuple[float, ...] = ()
    batch_size_grid: tuple[int, ...] = ()
    robust_step_size_grid: tuple[float, ...] = ()
    gamma_grid: tuple[float, ...] = ()
    finetune_summary: str = ""
    skip_finetune: bool = False
    finetune_only: bool = False
    skip_baseline: bool = False
    baseline_only: bool = False


class GroupDROModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.encoder, self.feat_dim = build_backbone(backbone_name, pretrained)
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class GroupDROLossComputer:
    def __init__(
        self,
        n_groups: int,
        step_size: float = 0.01,
        gamma: float = 0.1,
        adj: float = 0.0,
        device: torch.device = torch.device("cuda"),
    ):
        self.n_groups = n_groups
        self.step_size = step_size
        self.gamma = gamma
        self.adv_probs = torch.ones(n_groups, device=device) / n_groups
        self.exp_avg_loss = torch.zeros(n_groups, device=device)
        self.exp_avg_initialized = torch.zeros(n_groups, dtype=torch.bool, device=device)
        self.adj = torch.full((n_groups,), adj, device=device)
        self.device = device

    def compute_group_avg(
        self, losses: torch.Tensor, group_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_map = (
            group_idx.unsqueeze(0) == torch.arange(self.n_groups, device=self.device).unsqueeze(1)
        ).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss: torch.Tensor, group_count: torch.Tensor) -> None:
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * self.exp_avg_initialized.float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = self.exp_avg_initialized | (group_count > 0)

    def loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        sens_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_per = F.binary_cross_entropy_with_logits(outputs, labels, reduction="none")
        loss_flat, g_flat = flatten_multilabel(loss_per, sens_ids)
        lab_flat, _ = flatten_multilabel(labels.long(), sens_ids)
        joint_ids = build_joint_group_ids(lab_flat, g_flat, n_classes=2)

        group_loss, group_count = self.compute_group_avg(loss_flat, joint_ids)
        preds = (torch.sigmoid(outputs) >= 0.5).long()
        acc_per = (preds == labels.long()).float()
        acc_flat, _ = flatten_multilabel(acc_per, sens_ids)
        group_acc, _ = self.compute_group_avg(acc_flat, joint_ids)

        self.update_exp_avg_loss(group_loss, group_count)
        adjusted_loss = group_loss
        if self.adj.sum() > 0:
            adjusted_loss = adjusted_loss + self.adj
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / self.adv_probs.sum()
        robust_loss = group_loss @ self.adv_probs
        return robust_loss / NUM_CLASSES, group_loss, group_acc


def train_model(
    model: GroupDROModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    baseline_report: dict[str, float] | None,
) -> tuple[GroupDROModel, dict[str, float]]:
    n_sens = compute_n_groups(cfg.demographic_attrs)
    n_groups = n_sens * 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    loss_computer = GroupDROLossComputer(
        n_groups=n_groups,
        step_size=cfg.robust_step_size,
        gamma=cfg.gamma,
        adj=cfg.generalization_adjustment,
        device=device,
    )
    scaler = make_grad_scaler(cfg.use_amp, device)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_report: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            sens_ids = build_group_ids(batch, cfg.demographic_attrs, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(cfg.use_amp, device):
                outputs = model(inputs)
                outputs = outputs.float()
                robust_loss, _, _ = loss_computer.loss(outputs, labels, sens_ids)
            scaler.scale(robust_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += robust_loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{robust_loss.item():.4f}"})

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
        print(
            f"Epoch {epoch + 1:03d}: train_loss={avg_loss:.4f}, "
            f"macro_auc={val_report['overall_macro_auc']:.4f}, "
            f"{cfg.selection_metric}={score:.4f}"
        )
        scheduler.step(val_report["val_loss"])
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_report = dict(val_report)

    model.load_state_dict(best_state)
    if not best_report:
        best_report = evaluate_subgroup_report(model, val_loader, device, cfg.selection_attr)
        best_report["selection_score"] = compute_selection_score(
            best_report,
            selection_metric=cfg.selection_metric,
            selection_attr=cfg.selection_attr,
            baseline_report=baseline_report,
        )
    return model, best_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group DRO (Fusion-style finetune) for CheXpert (multi-label)."
    )
    add_common_arguments(
        parser,
        default_output_dir="chexpert_groupdro_outputs",
        default_backbone="resnet50",
        default_epochs=20,
        default_batch_size=128,
    )
    add_finetune_arguments(parser)
    parser.add_argument("--robust-step-size", type=float, default=Config.robust_step_size)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--generalization-adjustment", type=float, default=Config.generalization_adjustment)
    parser.add_argument("--robust-step-size-grid", type=str, default="0.001,0.01,0.1")
    parser.add_argument("--gamma-grid", type=str, default="0.01,0.1,1.0")
    args = parser.parse_args()

    cfg = Config()
    cfg.data_root = args.data_root
    cfg.output_dir = args.output_dir
    cfg.erm_hparams_dir = args.erm_hparams_dir
    cfg.backbone = args.backbone.lower()
    if cfg.backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone {cfg.backbone}")
    apply_demographic_config(cfg, args.demographic_subgroup, args.demographic_attrs)
    cfg.selection_metric = normalize_selection_metric(args.selection_metric)
    cfg.selection_attr = resolve_selection_attr(args.selection_attr, cfg.demographic_attrs)
    cfg.baseline_selection_attr = resolve_selection_attr(args.baseline_selection_attr, ("sex",))

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
    cfg.robust_step_size = args.robust_step_size
    cfg.gamma = args.gamma
    cfg.generalization_adjustment = args.generalization_adjustment
    cfg.pretrained = not args.no_pretrained
    cfg.use_amp = True
    if args.no_use_amp:
        cfg.use_amp = False
    elif args.use_amp:
        cfg.use_amp = True
    cfg.lr_grid = parse_float_grid(args.lr_grid)
    cfg.weight_decay_grid = parse_float_grid(args.weight_decay_grid)
    cfg.batch_size_grid = tuple(int(x) for x in parse_float_grid(args.batch_size_grid))
    cfg.robust_step_size_grid = parse_float_grid(args.robust_step_size_grid)
    cfg.gamma_grid = parse_float_grid(args.gamma_grid)
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

    if cfg.baseline_only or needs_local_baseline(cfg.selection_metric):
        ensure_baseline_erm(cfg, cfg.backbone, device)
    if cfg.baseline_only:
        print("baseline_only=1; skipping GroupDRO training.")
        return

    run_group = run_group_name("GroupDRO", cfg.demographic_attrs, cfg.demographic_subgroup)
    ckpt_root, results_root = get_output_roots(
        cfg.output_dir, "GroupDRO", cfg.demographic_attrs, cfg.demographic_subgroup
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
    mid_step = mid_of(cfg.robust_step_size_grid, cfg.robust_step_size)
    mid_gamma = mid_of(cfg.gamma_grid, cfg.gamma)
    initial_hparams: dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": float(mid_wd),
        "batch_size": int(mid_bs),
        "epochs": cfg.epochs,
        "robust_step_size": float(mid_step),
        "gamma": float(mid_gamma),
        "generalization_adjustment": cfg.generalization_adjustment,
    }

    def trial_fn(hparams: dict[str, Any], stage: str) -> dict[str, Any]:
        trial_cfg = apply_hparams_to_cfg(cfg, hparams)
        set_seed(int(cfg.seeds[0]))
        tr, va, _te = _loaders_for(int(trial_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(
            trial_cfg, trial_cfg.backbone, int(cfg.seeds[0]), va, device
        )
        print(f"\n--- Finetune [{stage}] hparams={hparams} ---")
        model = GroupDROModel(trial_cfg.backbone, pretrained=trial_cfg.pretrained).to(device)
        model, report = train_model(model, tr, va, device, trial_cfg, baseline_report)
        score = float(report.get("selection_score", float("nan")))
        return {"stage": stage, "hparams": dict(hparams), "metrics": report, "selection_score": score}

    stages = [
        ("lr", cfg.lr_grid),
        ("weight_decay", cfg.weight_decay_grid),
        ("batch_size", cfg.batch_size_grid),
        ("robust_step_size", cfg.robust_step_size_grid),
        ("gamma", cfg.gamma_grid),
    ]
    grids_meta = {
        "lr": list(cfg.lr_grid),
        "weight_decay": list(cfg.weight_decay_grid),
        "batch_size": list(cfg.batch_size_grid),
        "robust_step_size": list(cfg.robust_step_size_grid),
        "gamma": list(cfg.gamma_grid),
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
        print(f"GroupDRO | seed={seed} | backbone={train_cfg.backbone} | attrs={train_cfg.demographic_attrs}")
        set_seed(int(seed))
        tr, va, te = _loaders_for(int(train_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(train_cfg, train_cfg.backbone, int(seed), va, device)
        model = GroupDROModel(train_cfg.backbone, pretrained=train_cfg.pretrained).to(device)
        model, best_report = train_model(model, tr, va, device, train_cfg, baseline_report)

        attr_suffix = "_".join(train_cfg.demographic_attrs)
        run_name = f"{train_cfg.backbone}_GroupDRO_{attr_suffix}"
        rows = collect_test_rows(model, te, device, run_name, int(seed), use_amp=train_cfg.use_amp)
        csv_path, ckpt_path = persist_seed_run(
            results_root,
            ckpt_root,
            int(seed),
            model.state_dict(),
            f"groupdro_{train_cfg.backbone}.pt",
            rows,
        )
        save_json(
            ckpt_path.parent / f"groupdro_{train_cfg.backbone}_val_metrics.json",
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

    print(f"\nGroup DRO complete → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
