from __future__ import annotations

import argparse
import copy
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from chexpert_fairness_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SELECTION_METRIC,
    EVAL_CLASS_INDICES,
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
    macro_auc_from_logits,
    make_grad_scaler,
    maybe_load_baseline_report,
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
)


@dataclass
class Config:
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = "chexpert_fairsde_outputs"
    erm_hparams_dir: str = ""
    backbone: str = "resnet50"
    batch_size: int = 128
    epochs: int = 20
    erm_epochs: int = 20
    sde_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224
    seeds: tuple[int, ...] = ()
    demographic_subgroup: str | None = None
    demographic_attrs: tuple[str, ...] = ("sex",)
    selection_attr: str = "sex"
    baseline_selection_attr: str = "sex"
    pretrained: bool = True
    lambda1: float = 0.1
    lambda2: float = 1.0
    lambda3: float = 1.0
    temperature: float = 0.07
    # Internal-only: FairSDE Stage-1/Stage-2 expert training criterion.
    expert_selection_metric: str = "overall_macro_auc"
    # Outer Fusion checkpoint/finetune selection metric (default overall_macro_auc).
    selection_metric: str = DEFAULT_SELECTION_METRIC
    use_amp: bool = True
    lr_grid: tuple[float, ...] = ()
    weight_decay_grid: tuple[float, ...] = ()
    batch_size_grid: tuple[int, ...] = ()
    lambda1_grid: tuple[float, ...] = ()
    temperature_grid: tuple[float, ...] = ()
    finetune_summary: str = ""
    skip_finetune: bool = False
    finetune_only: bool = False
    skip_baseline: bool = False
    baseline_only: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Models (unchanged from CheXpert FairSDE algorithm; multilabel outputs)
# ─────────────────────────────────────────────────────────────────────────────

class ERMModel(nn.Module):
    """FairSDE's internal Stage-1 expert. Distinct from the Fusion
    ``ImageOnlyERM`` local baseline (used only for outer-protocol selection
    scoring, e.g. min_improvement); this model is retrained fresh inside
    every outer trial/seed as part of the FairSDE algorithm itself."""

    def __init__(self, backbone_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class VirtualCenter(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.center = Parameter(torch.Tensor(feat_dim, num_classes))
        self.center.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)


class FairSDEModel(nn.Module):
    def __init__(
        self, backbone_name: str, num_groups: int, num_classes: int = NUM_CLASSES, pretrained: bool = True
    ):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.decoupled_classifiers = nn.ModuleDict({
            str(i): nn.Linear(self.feat_dim, num_classes) for i in range(num_groups)
        })
        self.virtual_centers = nn.ModuleDict({
            str(i): VirtualCenter(self.feat_dim, num_classes) for i in range(num_groups)
        })
        self.attr_classifier = nn.Linear(self.feat_dim, num_groups)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.backbone(x)
        outputs, vcenters = [], []
        norm_feat = F.normalize(feature, dim=1)
        for i in range(self.num_groups):
            outputs.append(self.decoupled_classifiers[str(i)](feature))
            vcenters.append(norm_feat.mm(F.normalize(self.virtual_centers[str(i)].center, dim=0)))
        outputs = torch.stack(outputs)
        vcenters = torch.stack(vcenters)
        attr_pred = self.attr_classifier(feature)
        return outputs, vcenters, feature, attr_pred

    def predict_for_groups(self, x: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        outputs, _, _, _ = self.forward(x)
        b = x.size(0)
        return outputs[group_ids, torch.arange(b, device=outputs.device)]


class FairSDECombinedModel(nn.Module):
    """Wraps the ERM + FairSDE experts so the post-selection combined model
    can be scored with ``evaluate_subgroup_report`` for the Fusion outer
    protocol."""

    def __init__(self, erm_model: ERMModel, sde_model: FairSDEModel):
        super().__init__()
        self.erm_model = erm_model
        self.sde_model = sde_model


def make_combined_forward_fn(
    demo_attrs: tuple[str, ...], use_erm: np.ndarray,
) -> Callable[[nn.Module, dict, torch.device], torch.Tensor]:
    def _forward(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
        x = batch["image"].to(device, non_blocking=True)
        gids = build_group_ids(batch, demo_attrs, device)
        erm_logits = model.erm_model(x).float()
        sde_logits = model.sde_model.predict_for_groups(x, gids).float()
        use_erm_mask = torch.tensor(
            [use_erm[g.item()] == 1 for g in gids], dtype=torch.bool, device=device
        )
        return torch.where(use_erm_mask.unsqueeze(1), erm_logits, sde_logits)

    return _forward


# ─────────────────────────────────────────────────────────────────────────────
# Internal expert evaluation (macro AUC over 5 eval pathologies, or val_loss)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_expert(
    forward_fn: Callable[[dict], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    metric: str,
) -> float:
    logits_all, labels_all = [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        y = batch["labels"].to(device, non_blocking=True)
        with autocast_ctx(use_amp, device):
            logits = forward_fn(batch)
        logits = logits.float()
        loss = F.binary_cross_entropy_with_logits(logits, y)
        bs = y.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs
        logits_all.append(logits.cpu())
        labels_all.append(y.cpu())
    if n == 0:
        return float("nan")
    if metric == "val_loss":
        return -(loss_sum / n)  # negate so "higher is better" holds uniformly
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return macro_auc_from_logits(logits, labels)


def train_erm(
    model: ERMModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> ERMModel:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_grad_scaler(cfg.use_amp, device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")

    for epoch in range(cfg.erm_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"ERM {epoch + 1}/{cfg.erm_epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(cfg.use_amp, device):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        score = evaluate_expert(
            lambda b: model(b["image"].to(device, non_blocking=True)),
            val_loader, device, cfg.use_amp, cfg.expert_selection_metric,
        )
        print(f"ERM Epoch {epoch + 1:03d}: {cfg.expert_selection_metric}={score:.4f}")
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def train_fairsde(
    model: FairSDEModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> FairSDEModel:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_grad_scaler(cfg.use_amp, device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    num_groups = model.num_groups

    for epoch in range(cfg.sde_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"SDE {epoch + 1}/{cfg.sde_epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            group_ids = build_group_ids(batch, cfg.demographic_attrs, device)
            b = x.size(0)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(cfg.use_amp, device):
                outputs, vcenters, feature, attr_pred = model(x)
            outputs = outputs.float()
            vcenters = vcenters.float()
            feature = feature.float()
            attr_pred = attr_pred.float()

            loss_disc = F.cross_entropy(attr_pred, group_ids)

            loss_decouple = torch.tensor(0.0, device=device)
            for i in range(num_groups):
                mask = (group_ids == i).nonzero(as_tuple=True)[0]
                if mask.numel() > 0:
                    loss_decouple = loss_decouple + F.binary_cross_entropy_with_logits(
                        outputs[i][mask], y[mask]
                    )

            loss_virt = torch.tensor(0.0, device=device)
            vc_combined = torch.zeros(b, model.num_classes, device=device)
            for i in range(num_groups):
                mask = (group_ids == i).nonzero(as_tuple=True)[0]
                if mask.numel() == 0:
                    continue
                for j in range(model.num_classes):
                    logits_j = outputs[i][mask, j] / cfg.temperature
                    vc_j = vcenters[i][mask, j] / cfg.temperature
                    softmax_sim = F.softmax(torch.stack([logits_j, vc_j], dim=1), dim=1)[:, 0]
                    target = y[mask, j]
                    loss_virt = loss_virt - (
                        target * torch.log(softmax_sim + 1e-8)
                        + (1 - target) * torch.log(1 - softmax_sim + 1e-8)
                    ).mean() / model.num_classes
                vc_combined[mask] = vcenters[i][mask]

            z_i = F.normalize(feature, dim=1)
            same_group = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
            same_label = (y.unsqueeze(0) == y.unsqueeze(1)).all(dim=2)
            mask_pos = same_group & same_label
            mask_neg_cross = (~same_group) & same_label

            pos_indices = torch.full((b,), -1, dtype=torch.long, device=device)
            neg_indices = torch.full((b,), -1, dtype=torch.long, device=device)
            for i in range(b):
                pos_cands = mask_pos[i].clone()
                pos_cands[i] = False
                if pos_cands.any():
                    pos_indices[i] = pos_cands.nonzero(as_tuple=True)[0][
                        torch.randint(pos_cands.sum(), (1,))
                    ]
                else:
                    pos_indices[i] = i
                neg_cands = mask_neg_cross[i]
                if neg_cands.any():
                    neg_indices[i] = neg_cands.nonzero(as_tuple=True)[0][
                        torch.randint(neg_cands.sum(), (1,))
                    ]
                else:
                    neg_indices[i] = 0

            exp_pos = torch.exp((z_i * z_i[pos_indices]).sum(dim=1) / cfg.temperature)
            exp_pos = exp_pos + torch.exp((y * vc_combined).sum(dim=1) / cfg.temperature)
            vc_perm = vcenters.permute(1, 0, 2)
            neg_mask = torch.ones(b, num_groups, model.num_classes, dtype=torch.bool, device=device)
            for bi in range(b):
                neg_mask[bi, group_ids[bi], :] = False
            masked_vc = vc_perm[neg_mask].view(b, -1)
            exp_neg = torch.exp(masked_vc / cfg.temperature).sum(dim=1).clamp(min=1.0)
            exp_neg_cross = torch.exp((z_i * z_i[neg_indices]).sum(dim=1) / cfg.temperature)
            loss_div = -torch.log(exp_pos / (exp_neg_cross + exp_neg + 1e-8)).mean()

            loss = loss_decouple + cfg.lambda1 * loss_disc + cfg.lambda2 * loss_virt + cfg.lambda3 * loss_div
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(dec=f"{loss_decouple.item():.3f}")

        model.eval()
        score = evaluate_expert(
            lambda b: model.predict_for_groups(
                b["image"].to(device, non_blocking=True),
                build_group_ids(b, cfg.demographic_attrs, device),
            ),
            val_loader, device, cfg.use_amp, cfg.expert_selection_metric,
        )
        print(f"SDE Epoch {epoch + 1:03d}: {cfg.expert_selection_metric}={score:.4f}")
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: SDE Selection (brute-force over group assignments, macro AUC)
# ─────────────────────────────────────────────────────────────────────────────

def macro_auc_eval(probs: np.ndarray, labels: np.ndarray) -> float:
    aucs = []
    for j in EVAL_CLASS_INDICES:
        yj = labels[:, j]
        if len(np.unique(yj)) < 2:
            continue
        try:
            from sklearn.metrics import roc_auc_score

            aucs.append(float(roc_auc_score(yj, probs[:, j])))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.5


@torch.no_grad()
def collect_val_probs(
    sde_model: FairSDEModel,
    erm_model: ERMModel,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> dict[str, np.ndarray]:
    sde_model.eval()
    erm_model.eval()
    sde_probs, erm_probs, labels, groups = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["labels"].numpy()
        gids = build_group_ids(batch, cfg.demographic_attrs, device)
        with autocast_ctx(cfg.use_amp, device):
            erm_logits = erm_model(x)
            sde_logits = sde_model.predict_for_groups(x, gids)
        erm_p = torch.sigmoid(erm_logits.float()).cpu().numpy()
        sde_p = torch.sigmoid(sde_logits.float()).cpu().numpy()
        sde_probs.append(sde_p)
        erm_probs.append(erm_p)
        labels.append(y)
        groups.append(gids.cpu().numpy())
    return {
        "sde_probs": np.concatenate(sde_probs),
        "erm_probs": np.concatenate(erm_probs),
        "labels": np.concatenate(labels),
        "groups": np.concatenate(groups),
    }


def sde_select(val_data: dict[str, np.ndarray], num_groups: int) -> np.ndarray:
    """Brute-force search over 2^num_groups assignments (macro AUC over the 5
    competition pathologies). use_erm[g]=1 means group g uses ERM; 0 means
    FairSDE."""
    labels = val_data["labels"]
    groups = val_data["groups"]
    sde_probs = val_data["sde_probs"]
    erm_probs = val_data["erm_probs"]

    def group_macro_aucs(probs: np.ndarray) -> list[float]:
        aucs = []
        for g in range(num_groups):
            mask = groups == g
            if mask.sum() < 10:
                aucs.append(0.5)
            else:
                aucs.append(macro_auc_eval(probs[mask], labels[mask]))
        return aucs

    erm_group = group_macro_aucs(erm_probs)
    erm_gap = max(erm_group) - min(erm_group)
    best_assignment = np.zeros(num_groups, dtype=int)
    best_overall = -1.0

    for bits in itertools.product([0, 1], repeat=num_groups):
        assignment = np.array(bits, dtype=int)
        combined = np.copy(sde_probs)
        for g in range(num_groups):
            if assignment[g] == 1:
                combined[groups == g] = erm_probs[groups == g]
        g_aucs = group_macro_aucs(combined)
        gap = max(g_aucs) - min(g_aucs)
        valid = all(g_aucs[g] >= erm_group[g] - 1e-6 for g in range(num_groups))
        valid = valid and gap <= erm_gap + 1e-6
        if valid:
            overall = macro_auc_eval(combined, labels)
            if overall > best_overall:
                best_overall = overall
                best_assignment = assignment.copy()
    return best_assignment


def _group_labels(demo_attrs: tuple[str, ...], num_groups: int) -> list[str]:
    if len(demo_attrs) == 1 and demo_attrs[0] == "sex":
        return ["Female", "Male"]
    if len(demo_attrs) == 1 and demo_attrs[0] == "age":
        return ["<=60", ">60"]
    return [f"group_{i}" for i in range(num_groups)]


# ─────────────────────────────────────────────────────────────────────────────
# Fusion outer protocol: run the (unchanged) internal FairSDE pipeline once
# per outer trial/seed, then score the final combined model.
# ─────────────────────────────────────────────────────────────────────────────

def run_fairsde_pipeline(
    cfg: Config,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[ERMModel, FairSDEModel, np.ndarray, int]:
    num_groups = compute_n_groups(cfg.demographic_attrs)
    set_seed(int(seed))

    print("\n--- Stage 1: ERM Expert Training (internal) ---")
    erm_model = ERMModel(cfg.backbone, pretrained=cfg.pretrained).to(device)
    erm_model = train_erm(erm_model, train_loader, val_loader, device, cfg)

    print("\n--- Stage 2: FairSDE Expert Training ---")
    sde_model = FairSDEModel(cfg.backbone, num_groups=num_groups, pretrained=cfg.pretrained).to(device)
    sde_model = train_fairsde(sde_model, train_loader, val_loader, device, cfg)

    print("\n--- Stage 3: SDE Selection (validation, macro AUC) ---")
    val_data = collect_val_probs(sde_model, erm_model, val_loader, device, cfg)
    use_erm = sde_select(val_data, num_groups)
    for g, label in enumerate(_group_labels(cfg.demographic_attrs, num_groups)):
        expert = "ERM" if use_erm[g] else "FairSDE"
        print(f"  {label}: use {expert}")

    return erm_model, sde_model, use_erm, num_groups


def score_combined_model(
    erm_model: ERMModel,
    sde_model: FairSDEModel,
    use_erm: np.ndarray,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    baseline_report: dict[str, float] | None,
) -> dict[str, float]:
    combined = FairSDECombinedModel(erm_model, sde_model).to(device)
    forward_fn = make_combined_forward_fn(cfg.demographic_attrs, use_erm)
    report = evaluate_subgroup_report(
        combined, val_loader, device, cfg.selection_attr, use_amp=False, forward_fn=forward_fn
    )
    score = compute_selection_score(
        report,
        selection_metric=cfg.selection_metric,
        selection_attr=cfg.selection_attr,
        baseline_report=baseline_report,
    )
    report["selection_score"] = float(score) if not math.isnan(score) else float("nan")
    return report


def main() -> None:
    default_out = str(Path(__file__).resolve().parent / "chexpert_fairsde_outputs")
    parser = argparse.ArgumentParser(
        description="FairSDE (Fusion-style finetune) for CheXpert (multi-label)."
    )
    add_common_arguments(
        parser,
        default_output_dir=default_out,
        default_backbone="resnet50",
        default_epochs=20,
        default_batch_size=128,
    )
    add_finetune_arguments(parser)
    parser.add_argument("--erm-epochs", type=int, default=None)
    parser.add_argument("--sde-epochs", type=int, default=None)
    parser.add_argument("--lambda1", type=float, default=Config.lambda1, help="Weight for L_disc")
    parser.add_argument("--lambda2", type=float, default=Config.lambda2, help="Weight for L_virt")
    parser.add_argument("--lambda3", type=float, default=Config.lambda3, help="Weight for L_div")
    parser.add_argument("--temperature", type=float, default=Config.temperature)
    parser.add_argument(
        "--expert-selection-metric",
        type=str,
        choices=["overall_macro_auc", "val_loss"],
        default=Config.expert_selection_metric,
        help=(
            "Internal-only: FairSDE Stage-1/Stage-2 expert training criterion "
            "(macro AUC over the 5 competition pathologies, or val_loss). "
            "Distinct from --selection-metric (outer Fusion checkpoint/finetune)."
        ),
    )
    parser.add_argument("--lambda1-grid", type=str, default="0.01,0.1,1.0")
    parser.add_argument("--temperature-grid", type=str, default="0.07,0.1,0.2")
    args = parser.parse_args()

    cfg = Config()
    cfg.data_root = args.data_root
    cfg.output_dir = args.output_dir
    cfg.erm_hparams_dir = args.erm_hparams_dir
    cfg.backbone = args.backbone.lower()
    if cfg.backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone '{cfg.backbone}'. Choose from: {SUPPORTED_BACKBONES}")
    attr_suffix = apply_demographic_config(cfg, args.demographic_subgroup, args.demographic_attrs)
    cfg.selection_metric = normalize_selection_metric(args.selection_metric)
    cfg.selection_attr = resolve_selection_attr(args.selection_attr, cfg.demographic_attrs)
    cfg.baseline_selection_attr = resolve_selection_attr(args.baseline_selection_attr, ("sex",))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_file = Path(args.seeds_file) if str(args.seeds_file).strip() else (out_dir / "experiment_seeds.json")
    cfg.seeds = resolve_seeds(
        args.seeds, args.num_random_seeds, seeds_file if seeds_file.is_file() else None, seeds_file
    )

    cfg.epochs = args.epochs
    cfg.erm_epochs = args.erm_epochs if args.erm_epochs is not None else args.epochs
    cfg.sde_epochs = args.sde_epochs if args.sde_epochs is not None else args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers
    cfg.image_size = args.image_size
    cfg.pretrained = not args.no_pretrained
    cfg.lambda1 = args.lambda1
    cfg.lambda2 = args.lambda2
    cfg.lambda3 = args.lambda3
    cfg.temperature = args.temperature
    cfg.expert_selection_metric = args.expert_selection_metric

    cfg.use_amp = True
    if args.no_use_amp:
        cfg.use_amp = False
    elif args.use_amp:
        cfg.use_amp = True

    cfg.lr_grid = parse_float_grid(args.lr_grid)
    cfg.weight_decay_grid = parse_float_grid(args.weight_decay_grid)
    cfg.batch_size_grid = tuple(int(x) for x in parse_float_grid(args.batch_size_grid))
    cfg.lambda1_grid = parse_float_grid(args.lambda1_grid)
    cfg.temperature_grid = parse_float_grid(args.temperature_grid)
    cfg.finetune_summary = args.finetune_summary
    cfg.skip_finetune = bool(args.skip_finetune)
    cfg.finetune_only = bool(args.finetune_only)
    cfg.skip_baseline = bool(args.skip_baseline)
    cfg.baseline_only = bool(args.baseline_only)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_groups = compute_n_groups(cfg.demographic_attrs)

    print(f"Using device: {device}")
    print(f"Backbone: {cfg.backbone} (pretrained={cfg.pretrained})")
    print(f"Seeds: {cfg.seeds}")
    print(f"ERM hparams dir: {cfg.erm_hparams_dir or '(REQUIRED)'}")
    print(f"Selection metric: {cfg.selection_metric}")
    print(f"Expert selection metric (internal): {cfg.expert_selection_metric}")
    print(f"Baseline selection_attr={cfg.baseline_selection_attr}")
    print(f"Demographic: {cfg.demographic_subgroup or attr_suffix} -> {cfg.demographic_attrs} ({num_groups} groups)")
    print(f"ERM epochs: {cfg.erm_epochs}, SDE epochs: {cfg.sde_epochs}")
    print(f"Lambdas: l1={cfg.lambda1}, l2={cfg.lambda2}, l3={cfg.lambda3}, tau={cfg.temperature}")
    print(f"AMP: {cfg.use_amp}")

    if cfg.baseline_only or needs_local_baseline(cfg.selection_metric):
        ensure_baseline_erm(cfg, cfg.backbone, device)
    if cfg.baseline_only:
        print("baseline_only=1; skipping FairSDE training.")
        return

    run_group = run_group_name("FairSDE", cfg.demographic_attrs, cfg.demographic_subgroup)
    ckpt_root, results_root = get_output_roots(
        cfg.output_dir, "FairSDE", cfg.demographic_attrs, cfg.demographic_subgroup
    )
    ckpt_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    print(f"Demographic attrs: {cfg.demographic_attrs} | selection_attr={cfg.selection_attr}")
    print(f"Checkpoint metric: {cfg.selection_metric} (final combined model)")

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
    mid_lambda1 = mid_of(cfg.lambda1_grid, cfg.lambda1)
    mid_temperature = mid_of(cfg.temperature_grid, cfg.temperature)
    initial_hparams: dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": float(mid_wd),
        "batch_size": int(mid_bs),
        "erm_epochs": cfg.erm_epochs,
        "sde_epochs": cfg.sde_epochs,
        "lambda1": float(mid_lambda1),
        "temperature": float(mid_temperature),
    }

    def trial_fn(hparams: dict[str, Any], stage: str) -> dict[str, Any]:
        trial_cfg = apply_hparams_to_cfg(cfg, hparams)
        seed0 = int(cfg.seeds[0])
        tr, va, _te = _loaders_for(int(trial_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(trial_cfg, trial_cfg.backbone, seed0, va, device)
        print(f"\n--- Finetune [{stage}] hparams={hparams} ---")
        erm_model, sde_model, use_erm, _ng = run_fairsde_pipeline(trial_cfg, seed0, tr, va, device)
        report = score_combined_model(erm_model, sde_model, use_erm, va, device, trial_cfg, baseline_report)
        score = float(report.get("selection_score", float("nan")))
        return {"stage": stage, "hparams": dict(hparams), "metrics": report, "selection_score": score}

    stages = [
        ("lr", cfg.lr_grid),
        ("weight_decay", cfg.weight_decay_grid),
        ("batch_size", cfg.batch_size_grid),
        ("lambda1", cfg.lambda1_grid),
        ("temperature", cfg.temperature_grid),
    ]
    grids_meta = {
        "lr": list(cfg.lr_grid),
        "weight_decay": list(cfg.weight_decay_grid),
        "batch_size": list(cfg.batch_size_grid),
        "lambda1": list(cfg.lambda1_grid),
        "temperature": list(cfg.temperature_grid),
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
        print(f"FairSDE | seed={seed} | backbone={train_cfg.backbone} | attrs={train_cfg.demographic_attrs}")
        print("=" * 80)
        tr, va, te = _loaders_for(int(train_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(train_cfg, train_cfg.backbone, int(seed), va, device)
        erm_model, sde_model, use_erm, num_groups = run_fairsde_pipeline(train_cfg, seed, tr, va, device)

        seed_ckpt_dir = ckpt_root / f"seed_{seed}"
        seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(erm_model.state_dict(), seed_ckpt_dir / f"erm_{train_cfg.backbone}.pt")
        torch.save(sde_model.state_dict(), seed_ckpt_dir / f"fairsde_{train_cfg.backbone}.pt")
        np.save(seed_ckpt_dir / "use_erm.npy", use_erm)
        save_json(
            seed_ckpt_dir / "sde_selection.json",
            {"use_erm": use_erm.tolist(), "num_groups": num_groups},
        )
        print("Saved ERM + FairSDE checkpoints")

        val_report = score_combined_model(erm_model, sde_model, use_erm, va, device, train_cfg, baseline_report)
        print(
            f"Final combined model val: overall_macro_auc={val_report['overall_macro_auc']:.4f} "
            f"{train_cfg.selection_metric}={val_report['selection_score']:.4f}"
        )

        forward_fn = make_combined_forward_fn(train_cfg.demographic_attrs, use_erm)
        combined = FairSDECombinedModel(erm_model, sde_model).to(device)
        run_name = f"{train_cfg.backbone}_FairSDE_{attr_suffix}"
        rows = collect_test_rows(
            combined, te, device, run_name, int(seed), use_amp=False, forward_fn=forward_fn
        )
        csv_path = save_seed_results(results_root, int(seed), rows)
        save_json(
            seed_ckpt_dir / f"fairsde_{train_cfg.backbone}_val_metrics.json",
            {
                "seed": int(seed),
                "selection_metric": train_cfg.selection_metric,
                "selection_attr": train_cfg.selection_attr,
                "hparams": summary["hparams"],
                "val_metrics": val_report,
            },
        )
        print(f"Saved test predictions: {csv_path.resolve()}")

    print("\n" + "=" * 80)
    print("FairSDE training complete!")
    print(f"Results saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
