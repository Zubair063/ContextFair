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
from torch.autograd import Function
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fairvision_fairness_common import (
    ATTR_CARDINALITIES,
    DEFAULT_DATA_ROOT,
    DEFAULT_SELECTION_METRIC,
    add_common_arguments,
    add_finetune_arguments,
    amp_active,
    apply_demographic_config,
    apply_hparams_to_cfg,
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

# LAFTR only supports these backbones (validated in main).
LAFTR_BACKBONES = ("vgg16", "vit_b_16")
build_sensitive_ids = build_group_ids


@dataclass
class Config:
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = "fairvision_laftr_outputs"
    erm_hparams_dir: str = ""
    backbone: str = "vgg16"
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224
    seeds: tuple[int, ...] = ()
    demographic_subgroup: str | None = None
    demographic_attrs: tuple[str, ...] = ("gender",)
    selection_metric: str = DEFAULT_SELECTION_METRIC
    selection_attr: str = "gender"
    baseline_selection_attr: str = "gender"
    pretrained: bool = True
    use_amp: bool = True
    lr_grid: tuple[float, ...] = ()
    weight_decay_grid: tuple[float, ...] = ()
    batch_size_grid: tuple[int, ...] = ()
    fair_coeff_grid: tuple[float, ...] = ()
    zdim_grid: tuple[int, ...] = ()
    finetune_summary: str = ""
    skip_finetune: bool = False
    finetune_only: bool = False
    skip_baseline: bool = False
    baseline_only: bool = False
    # LAFTR-specific (discriminative LRs derived from cfg.lr; see
    # apply_backbone_lr_defaults)
    lr_backbone: float | None = None
    lr_head: float | None = None
    adv_lr: float | None = None
    zdim: int = 128
    fair_coeff: float = 0.1
    class_coeff: float = 1.0
    grl_lambda: float = 1.0
    aud_steps: int = 1
    adv_hidden_dim: int = 256
    dropout: float = 0.1
    grad_clip_norm: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Gradient reversal (encoder fools adversary)
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFn.apply(x, alpha)


# ─────────────────────────────────────────────────────────────────────────────
# LAFTR model
# ─────────────────────────────────────────────────────────────────────────────

class LAFTRModel(nn.Module):
    """Image encoder -> latent Z; classifier on Z; adversary on GRL(Z)."""

    def __init__(
        self,
        backbone_name: str,
        n_sensitive: int,
        zdim: int = 128,
        num_classes: int = 2,
        pretrained: bool = True,
        adv_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder, feat_dim = build_backbone(backbone_name, pretrained)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, zdim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(zdim, num_classes)
        self.adversary = nn.Sequential(
            nn.Linear(zdim, adv_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(adv_hidden_dim, n_sensitive),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(x))

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x))

    def backbone_parameters(self):
        return list(self.encoder.parameters())

    def head_parameters(self):
        return list(self.proj.parameters()) + list(self.classifier.parameters())

    def encoder_parameters(self):
        return self.backbone_parameters() + self.head_parameters()

    def adversary_parameters(self):
        return list(self.adversary.parameters())


def apply_backbone_lr_defaults(cfg: Config) -> None:
    """Derive backbone/head/adversary LRs from the single swept ``cfg.lr``.

    vgg16: unified LR for all encoder-side params.
    vit_b_16: backbone LR held at 1/100 of head LR (matches the original
    1e-5 backbone / 1e-3 head ratio) so the discriminative-LR behavior is
    preserved while a single hyperparameter is swept by the Fusion finetune
    grid.
    """
    if cfg.backbone == "vit_b_16":
        cfg.lr_head = cfg.lr
        cfg.lr_backbone = cfg.lr * 0.01
    else:
        cfg.lr_backbone = cfg.lr
        cfg.lr_head = cfg.lr
    cfg.adv_lr = cfg.lr_head


def build_enc_optimizer(model: LAFTRModel, cfg: Config) -> torch.optim.AdamW:
    """Discriminative LRs for ViT (backbone vs proj+classifier); unified LR for VGG16."""
    if cfg.backbone == "vit_b_16":
        param_groups = [
            {"params": model.backbone_parameters(), "lr": cfg.lr_backbone},
            {"params": model.head_parameters(), "lr": cfg.lr_head},
        ]
    else:
        param_groups = [{"params": model.encoder_parameters(), "lr": cfg.lr_backbone}]
    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _safe_loss_value(loss: torch.Tensor) -> float:
    v = float(loss.item())
    return v if np.isfinite(v) else float("nan")


def _optimizer_step_with_clip(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    parameters: list[torch.nn.Parameter],
    max_norm: float,
    amp_on: bool,
) -> bool:
    """Returns False if gradients are non-finite (step skipped)."""
    if amp_on:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        if amp_on:
            scaler.update()
        return False
    if amp_on:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return True


def train_batch(
    model: LAFTRModel,
    batch: dict[str, Any],
    device: torch.device,
    demo_attrs: tuple[str, ...],
    enc_optimizer: torch.optim.Optimizer,
    adv_optimizer: torch.optim.Optimizer,
    cfg: Config,
    enc_scaler: torch.amp.GradScaler,
    adv_scaler: torch.amp.GradScaler,
    amp_on: bool,
) -> tuple[float, float, float]:
    """One encode per batch: update adversary on Z.detach(), then encoder via GRL."""
    model.train()
    x = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    sensitive = build_sensitive_ids(batch, demo_attrs, device)

    enc_optimizer.zero_grad(set_to_none=True)
    adv_optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=amp_on):
        z = model.encode(x)

    # Adversary step(s) on detached latents
    adv_loss = torch.tensor(0.0, device=device)
    for _ in range(max(cfg.aud_steps, 1)):
        adv_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_on):
            adv_logits = model.adversary(z.detach())
            adv_loss = F.cross_entropy(adv_logits, sensitive)
        adv_scaler.scale(adv_loss).backward()
        _optimizer_step_with_clip(
            adv_optimizer, adv_scaler, model.adversary_parameters(), cfg.grad_clip_norm, amp_on,
        )

    # Encoder + classifier (GRL through shared adversary)
    with torch.amp.autocast("cuda", enabled=amp_on):
        cls_logits = model.classify(z)
        cls_loss = F.cross_entropy(cls_logits, labels)
        z_grl = gradient_reversal(z, cfg.grl_lambda)
        fair_logits = model.adversary(z_grl)
        fair_loss = F.cross_entropy(fair_logits, sensitive)
        total_loss = cfg.class_coeff * cls_loss + cfg.fair_coeff * fair_loss

    if not torch.isfinite(total_loss):
        enc_optimizer.zero_grad(set_to_none=True)
        return float("nan"), float("nan"), _safe_loss_value(adv_loss)

    enc_scaler.scale(total_loss).backward()
    _optimizer_step_with_clip(
        enc_optimizer, enc_scaler, model.encoder_parameters(), cfg.grad_clip_norm, amp_on,
    )

    return _safe_loss_value(cls_loss), _safe_loss_value(fair_loss), _safe_loss_value(adv_loss)


def train_model(
    model: LAFTRModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    baseline_report: dict[str, float] | None,
) -> tuple[LAFTRModel, dict[str, float]]:
    enc_optimizer = build_enc_optimizer(model, cfg)
    adv_optimizer = torch.optim.AdamW(
        model.adversary_parameters(),
        lr=cfg.adv_lr,
        weight_decay=cfg.weight_decay,
    )
    enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        enc_optimizer, T_max=cfg.epochs,
    )
    amp_on = amp_active(device, cfg.use_amp)
    enc_scaler = make_grad_scaler(cfg.use_amp, device)
    adv_scaler = make_grad_scaler(cfg.use_amp, device)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_report: dict[str, float] = {}

    for epoch in range(cfg.epochs):
        running_cls = 0.0
        running_fair = 0.0
        running_adv = 0.0
        n_batches = 0
        nan_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            cls_loss, fair_loss, adv_loss = train_batch(
                model, batch, device, cfg.demographic_attrs,
                enc_optimizer, adv_optimizer, cfg, enc_scaler, adv_scaler, amp_on,
            )
            if not np.isfinite(cls_loss):
                nan_batches += 1
                if nan_batches == 1:
                    print("  [WARN] Non-finite loss detected; consider lower --fair-coeff or --no-use-amp")
                continue
            running_cls += cls_loss
            running_fair += fair_loss
            running_adv += adv_loss
            n_batches += 1
            pbar.set_postfix({
                "cls": f"{cls_loss:.4f}",
                "fair": f"{fair_loss:.4f}",
                "adv": f"{adv_loss:.4f}",
            })

        enc_scheduler.step()
        val_report = evaluate_subgroup_report(
            model, val_loader, device, cfg.selection_attr, use_amp=amp_on
        )
        score = compute_selection_score(
            val_report,
            selection_metric=cfg.selection_metric,
            selection_attr=cfg.selection_attr,
            baseline_report=baseline_report,
        )
        val_report["selection_score"] = float(score) if not math.isnan(score) else float("nan")
        avg_cls = running_cls / max(n_batches, 1)
        avg_fair = running_fair / max(n_batches, 1)
        avg_adv = running_adv / max(n_batches, 1)
        print(
            f"Epoch {epoch + 1:03d}: cls_loss={avg_cls:.4f}, fair_loss={avg_fair:.4f}, "
            f"adv_loss={avg_adv:.4f}, overall_auc={val_report['overall_auc']:.4f}, "
            f"{cfg.selection_metric}={score:.4f}"
        )

        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_report = dict(val_report)

    model.load_state_dict(best_state)
    if not best_report:
        best_report = evaluate_subgroup_report(
            model, val_loader, device, cfg.selection_attr, use_amp=amp_on
        )
        best_report["selection_score"] = compute_selection_score(
            best_report,
            selection_metric=cfg.selection_metric,
            selection_attr=cfg.selection_attr,
            baseline_report=baseline_report,
        )
    return model, best_report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LAFTR (Fusion-style finetune) for FairVision Glaucoma."
    )
    add_common_arguments(
        parser,
        default_output_dir="fairvision_laftr_outputs",
        default_backbone="vgg16",
        default_epochs=20,
        default_batch_size=32,
    )
    add_finetune_arguments(parser)
    parser.add_argument("--zdim", type=int, default=Config.zdim,
                        help="Latent representation dimension")
    parser.add_argument("--fair-coeff", type=float, default=Config.fair_coeff,
                        help="Weight on adversarial fairness loss (default 0.1; 1.0 often causes NaN)")
    parser.add_argument("--class-coeff", type=float, default=Config.class_coeff,
                        help="Weight on classification loss")
    parser.add_argument("--grl-lambda", type=float, default=Config.grl_lambda,
                        help="Gradient reversal strength")
    parser.add_argument("--aud-steps", type=int, default=Config.aud_steps,
                        help="Adversary update steps per batch")
    parser.add_argument("--adv-hidden-dim", type=int, default=Config.adv_hidden_dim)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--grad-clip-norm", type=float, default=Config.grad_clip_norm,
                        help="Max gradient norm for encoder and adversary")
    parser.add_argument("--fair-coeff-grid", type=str, default="0.01,0.1,1.0")
    parser.add_argument("--zdim-grid", type=str, default="64,128,256")
    args = parser.parse_args()

    cfg = Config()
    cfg.data_root = args.data_root
    cfg.output_dir = args.output_dir
    cfg.erm_hparams_dir = args.erm_hparams_dir
    cfg.backbone = args.backbone.lower()
    if cfg.backbone not in LAFTR_BACKBONES:
        raise ValueError(
            f"LAFTR backbone must be one of {LAFTR_BACKBONES}, got '{cfg.backbone}'"
        )
    apply_demographic_config(cfg, args.demographic_subgroup, args.demographic_attrs)
    cfg.selection_metric = normalize_selection_metric(args.selection_metric)
    cfg.selection_attr = resolve_selection_attr(args.selection_attr, cfg.demographic_attrs)
    cfg.baseline_selection_attr = resolve_selection_attr(
        args.baseline_selection_attr, ("gender",)
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
    cfg.pretrained = not args.no_pretrained
    cfg.use_amp = False if args.no_use_amp else True
    cfg.zdim = args.zdim
    cfg.fair_coeff = args.fair_coeff
    cfg.class_coeff = args.class_coeff
    cfg.grl_lambda = args.grl_lambda
    cfg.aud_steps = args.aud_steps
    cfg.adv_hidden_dim = args.adv_hidden_dim
    cfg.dropout = args.dropout
    cfg.grad_clip_norm = args.grad_clip_norm
    apply_backbone_lr_defaults(cfg)

    cfg.lr_grid = parse_float_grid(args.lr_grid)
    cfg.weight_decay_grid = parse_float_grid(args.weight_decay_grid)
    cfg.batch_size_grid = tuple(int(x) for x in parse_float_grid(args.batch_size_grid))
    cfg.fair_coeff_grid = parse_float_grid(args.fair_coeff_grid)
    cfg.zdim_grid = tuple(int(x) for x in parse_float_grid(args.zdim_grid))
    cfg.finetune_summary = args.finetune_summary
    cfg.skip_finetune = bool(args.skip_finetune)
    cfg.finetune_only = bool(args.finetune_only)
    cfg.skip_baseline = bool(args.skip_baseline)
    cfg.baseline_only = bool(args.baseline_only)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_sensitive = compute_n_groups(cfg.demographic_attrs)

    print(f"Device: {device}")
    print(f"AMP: {amp_active(device, cfg.use_amp)}")
    print(f"Backbone: {cfg.backbone} (pretrained={cfg.pretrained})")
    print(f"Seeds: {cfg.seeds}")
    print(f"ERM hparams dir: {cfg.erm_hparams_dir or '(REQUIRED)'}")
    print(f"Selection metric: {cfg.selection_metric}")
    print(f"Baseline selection_attr={cfg.baseline_selection_attr}")
    print(f"ATTR_CARDINALITIES: { {a: ATTR_CARDINALITIES[a] for a in cfg.demographic_attrs} }")

    if cfg.baseline_only or needs_local_baseline(cfg.selection_metric):
        ensure_baseline_erm(cfg, cfg.backbone, device)
    if cfg.baseline_only:
        print("baseline_only=1; skipping LAFTR training.")
        return

    run_group = run_group_name("LAFTR", cfg.demographic_attrs, cfg.demographic_subgroup)
    ckpt_root, results_root = get_output_roots(
        cfg.output_dir, "LAFTR", cfg.demographic_attrs, cfg.demographic_subgroup
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
    mid_fair = mid_of(cfg.fair_coeff_grid, cfg.fair_coeff)
    mid_zdim = mid_of(cfg.zdim_grid, cfg.zdim)
    initial_hparams: dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": float(mid_wd),
        "batch_size": int(mid_bs),
        "epochs": cfg.epochs,
        "fair_coeff": float(mid_fair),
        "zdim": int(mid_zdim),
    }

    def trial_fn(hparams: dict[str, Any], stage: str) -> dict[str, Any]:
        trial_cfg = apply_hparams_to_cfg(cfg, hparams)
        apply_backbone_lr_defaults(trial_cfg)
        set_seed(int(cfg.seeds[0]))
        tr, va, _te = _loaders_for(int(trial_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(
            trial_cfg, trial_cfg.backbone, int(cfg.seeds[0]), va, device
        )
        print(f"\n--- Finetune [{stage}] hparams={hparams} ---")
        model = LAFTRModel(
            trial_cfg.backbone,
            n_sensitive=n_sensitive,
            zdim=trial_cfg.zdim,
            pretrained=trial_cfg.pretrained,
            adv_hidden_dim=trial_cfg.adv_hidden_dim,
            dropout=trial_cfg.dropout,
        ).to(device)
        model, report = train_model(model, tr, va, device, trial_cfg, baseline_report)
        score = float(report.get("selection_score", float("nan")))
        return {"stage": stage, "hparams": dict(hparams), "metrics": report, "selection_score": score}

    stages = [
        ("lr", cfg.lr_grid),
        ("weight_decay", cfg.weight_decay_grid),
        ("batch_size", cfg.batch_size_grid),
        ("fair_coeff", cfg.fair_coeff_grid),
        ("zdim", cfg.zdim_grid),
    ]
    grids_meta = {
        "lr": list(cfg.lr_grid),
        "weight_decay": list(cfg.weight_decay_grid),
        "batch_size": list(cfg.batch_size_grid),
        "fair_coeff": list(cfg.fair_coeff_grid),
        "zdim": list(cfg.zdim_grid),
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
    apply_backbone_lr_defaults(train_cfg)
    train_loader, val_loader, test_loader = _loaders_for(int(train_cfg.batch_size))
    print(f"Training all seeds with hparams: {summary['hparams']}")

    for seed in cfg.seeds:
        print("=" * 80)
        print(f"LAFTR | seed={seed} | backbone={train_cfg.backbone} | attrs={train_cfg.demographic_attrs}")
        set_seed(int(seed))
        tr, va, te = _loaders_for(int(train_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(
            train_cfg, train_cfg.backbone, int(seed), va, device
        )
        model = LAFTRModel(
            train_cfg.backbone,
            n_sensitive=n_sensitive,
            zdim=train_cfg.zdim,
            pretrained=train_cfg.pretrained,
            adv_hidden_dim=train_cfg.adv_hidden_dim,
            dropout=train_cfg.dropout,
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Params: {trainable_params:,} trainable / {total_params:,} total")

        model, best_report = train_model(model, tr, va, device, train_cfg, baseline_report)

        attr_suffix = "_".join(train_cfg.demographic_attrs)
        run_name = f"{train_cfg.backbone}_LAFTR_{attr_suffix}"
        rows = collect_test_rows(model, te, device, run_name, int(seed), train_cfg.use_amp)
        csv_path, ckpt_path = persist_seed_run(
            results_root,
            ckpt_root,
            int(seed),
            model.state_dict(),
            f"laftr_{train_cfg.backbone}.pt",
            rows,
        )
        save_json(
            ckpt_path.parent / f"laftr_{train_cfg.backbone}_val_metrics.json",
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

    print(f"\nLAFTR complete → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
