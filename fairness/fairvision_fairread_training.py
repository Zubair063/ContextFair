from __future__ import annotations

import argparse
import copy
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.autograd import Function
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fairvision_fairness_common import (
    ATTR_CARDINALITIES,
    DEFAULT_DATA_ROOT,
    DEFAULT_SELECTION_METRIC,
    FAIRREAD_BACKBONE,
    SUPPORTED_BACKBONES,
    add_common_arguments,
    add_finetune_arguments,
    amp_active,
    apply_demographic_config,
    apply_hparams_to_cfg,
    autocast_ctx,
    build_dataloaders,
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
    parse_int_grid,
    resolve_seeds,
    resolve_selection_attr,
    run_group_name,
    save_json,
    save_seed_results,
    set_seed,
    setup_cudnn_benchmark,
)


def compute_sa_dim(demo_attrs: tuple[str, ...]) -> int:
    """Total input dimension for the SA encoder after encoding.
    Binary attributes (cardinality 2) use a single scalar.
    Multi-class attributes use one-hot vectors.
    """
    dim = 0
    for attr in demo_attrs:
        card = ATTR_CARDINALITIES[attr]
        dim += 1 if card == 2 else card
    return dim


@dataclass
class Config:
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = "fairvision_fairread_outputs"
    erm_hparams_dir: str = ""
    backbone: str = FAIRREAD_BACKBONE
    batch_size: int = 32
    stage1_epochs: int = 20
    stage2_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224
    seeds: tuple[int, ...] = ()
    demographic_subgroup: str | None = None
    demographic_attrs: tuple[str, ...] = ("gender", "race")
    selection_metric: str = DEFAULT_SELECTION_METRIC
    selection_attr: str = "gender"
    baseline_selection_attr: str = "gender"
    num_cascade_blocks: int = 2
    mean_std_dim: int = 256
    adv_alpha: float = 0.1
    dropout: float = 0.1
    conv_reduction: int = 8
    pretrained: bool = True
    use_amp: bool = True
    lr_grid: tuple[float, ...] = ()
    weight_decay_grid: tuple[float, ...] = ()
    batch_size_grid: tuple[int, ...] = ()
    adv_alpha_grid: tuple[float, ...] = ()
    mean_std_dim_grid: tuple[int, ...] = ()
    finetune_summary: str = ""
    skip_finetune: bool = False
    finetune_only: bool = False
    skip_baseline: bool = False
    baseline_only: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer (for adversarial disentanglement)
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFn.apply(x, alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone Feature Extractors (spatial output: B x feat_dim x H x W)
# ─────────────────────────────────────────────────────────────────────────────

class _ViTSpatialWrapper(nn.Module):
    """Wraps torchvision ViT to produce spatial feature maps from patch tokens."""

    def __init__(self, vit_model: nn.Module):
        super().__init__()
        self.conv_proj = vit_model.conv_proj
        self.encoder = vit_model.encoder
        self.class_token = vit_model.class_token
        self.seq_length = vit_model.seq_length
        self.hidden_dim = vit_model.hidden_dim
        pos_embedding = vit_model.encoder.pos_embedding
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.conv_proj(x)
        gh, gw = x.shape[2], x.shape[3]
        x = x.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)
        cls_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.encoder.ln(self.encoder.layers(x + self.encoder.dropout(torch.zeros_like(x))))
        patches = x[:, 1:, :]
        return patches.permute(0, 2, 1).reshape(B, self.hidden_dim, gh, gw)


class _ResNetFeatures(nn.Module):
    """Extracts spatial feature maps from ResNet (before avgpool/fc)."""

    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x


def build_feature_extractor(name: str) -> tuple[nn.Module, int]:
    """Returns (feature_extractor, feat_dim) producing spatial maps (B, feat_dim, H, W)."""
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
    if name == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        return m.features, 1024
    if name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        return _ResNetFeatures(m), 2048
    if name == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        return m.features, 1280
    if name == "vgg16":
        m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        return m.features, 512
    if name == "vit_b_16":
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        return _ViTSpatialWrapper(m), 768
    raise ValueError(f"Unsupported backbone '{name}'. Choose from: {SUPPORTED_BACKBONES}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Disentangled Model
# ─────────────────────────────────────────────────────────────────────────────

class DisentangledEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        demo_attrs: tuple[str, ...],
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.demo_attrs = demo_attrs
        self.features, self.feat_dim = build_feature_extractor(backbone_name)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self.adv_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.adv_heads = nn.ModuleDict()
        for attr in demo_attrs:
            card = ATTR_CARDINALITIES[attr]
            self.adv_heads[attr] = nn.Sequential(
                nn.Linear(self.feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, card),
            )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        # Non-inplace: feat is reused by classifier + GRL adversary; inplace+AMP can NaN.
        feat = F.relu(feat, inplace=False)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

    def get_adversary_pred(self, feat: torch.Tensor, grl_alpha: float = 1.0) -> dict[str, torch.Tensor]:
        feat_rev = gradient_reversal(feat, grl_alpha)
        pooled = self.adv_pool(feat_rev)
        return {attr: head(pooled) for attr, head in self.adv_heads.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Refusion Cascade Rescale Model (FairREAD core)
# ─────────────────────────────────────────────────────────────────────────────

class CascadeRescaleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mean_std_dim: int,
        reduction: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        mid_channels = in_channels // reduction

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.proj_to_rescale = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, mean_std_dim),
            nn.ReLU(inplace=True),
        )

        self.proj_from_rescale = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mean_std_dim, in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        rescale_vec = self.proj_to_rescale(x)
        rescale_vec = rescale_vec * std + mean
        rescale_vec = self.proj_from_rescale(rescale_vec)
        rescale_vec = rescale_vec.unsqueeze(-1).unsqueeze(-1)
        x_rescaled = x * rescale_vec
        out = self.conv_block(x_rescaled)
        return out


class FairREADModel(nn.Module):
    def __init__(
        self,
        disentangled_encoder: nn.Module,
        feat_dim: int,
        demo_attrs: tuple[str, ...],
        sa_dim: int,
        num_classes: int = 2,
        num_cascade_blocks: int = 2,
        mean_std_dim: int = 256,
        dropout: float = 0.1,
        conv_reduction: int = 8,
        adv_alpha: float = 0.1,
    ):
        super().__init__()
        self.encoder = disentangled_encoder
        self.demo_attrs = demo_attrs
        self.feat_dim = feat_dim
        self.adv_alpha = adv_alpha

        self.sa_encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(sa_dim, mean_std_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.mean_decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mean_std_dim // 2, mean_std_dim),
        )
        self.std_decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mean_std_dim // 2, mean_std_dim),
        )

        self.cascade_blocks = nn.ModuleList([
            CascadeRescaleBlock(self.feat_dim, self.feat_dim, mean_std_dim, conv_reduction, dropout)
            for _ in range(num_cascade_blocks)
        ])

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        self.adv_shared = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.feat_dim // 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feat_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim // 8, self.feat_dim // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.feat_dim // 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.adv_heads = nn.ModuleDict()
        for attr in demo_attrs:
            card = ATTR_CARDINALITIES[attr]
            self.adv_heads[attr] = nn.Linear(self.feat_dim // 8, card)

    def forward(self, x: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = F.relu(self.encoder(x), inplace=False)

        sa_feat = self.sa_encoder(sa)
        mean = self.mean_decoder(sa_feat)
        std = self.std_decoder(sa_feat)

        h = feat
        for block in self.cascade_blocks:
            h = block(h, mean, std)

        h = self.pooling(h).squeeze(-1).squeeze(-1)
        h = self.classifier_dropout(h)
        logits = self.classifier(h)
        return logits

    def forward_with_adv(
        self, x: torch.Tensor, sa: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            feat = F.relu(self.encoder(x), inplace=False)

        sa_feat = self.sa_encoder(sa)
        mean = self.mean_decoder(sa_feat)
        std = self.std_decoder(sa_feat)

        h = feat
        for block in self.cascade_blocks:
            h = block(h, mean, std)

        h = self.pooling(h).squeeze(-1).squeeze(-1)
        h = self.classifier_dropout(h)
        logits = self.classifier(h)

        shared = self.adv_shared(feat)
        adv_preds = {attr: head(shared) for attr, head in self.adv_heads.items()}
        return logits, adv_preds


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build sensitive attribute tensor from batch
# ─────────────────────────────────────────────────────────────────────────────

def build_sa_input(batch: dict, demo_attrs: tuple[str, ...], device: torch.device) -> torch.Tensor:
    """Builds a float tensor of shape (B, sa_dim) for the SA encoder."""
    parts = []
    for attr in demo_attrs:
        vals = batch[attr].to(device).long().clamp(min=0)
        card = ATTR_CARDINALITIES[attr]
        if card == 2:
            parts.append(vals.float().unsqueeze(1))
        else:
            parts.append(F.one_hot(vals, num_classes=card).float())
    return torch.cat(parts, dim=1)


def build_sa_labels(
    batch: dict, demo_attrs: tuple[str, ...], device: torch.device,
) -> dict[str, torch.Tensor]:
    """Builds per-attribute integer label tensors for adversary CrossEntropyLoss."""
    return {attr: batch[attr].to(device).long().clamp(min=0) for attr in demo_attrs}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 Training: Disentanglement
# ─────────────────────────────────────────────────────────────────────────────

def compute_adversary_loss(
    adv_preds: dict[str, torch.Tensor],
    sa_labels: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Average CrossEntropyLoss across all per-attribute adversary heads."""
    total = torch.tensor(0.0, device=next(iter(adv_preds.values())).device)
    for attr, logits in adv_preds.items():
        total = total + F.cross_entropy(logits, sa_labels[attr])
    return total / len(adv_preds)


def train_stage1(
    model: DisentangledEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> DisentangledEncoder:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_auc = float("-inf")
    scaler = make_grad_scaler(cfg.use_amp, device)
    skipped = 0

    for epoch in range(cfg.stage1_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage1 {epoch + 1}/{cfg.stage1_epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            sa_labels = build_sa_labels(batch, cfg.demographic_attrs, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(cfg.use_amp, device):
                logits, feat = model(x, return_features=True)
            logits_f = logits.float()
            feat_f = feat.float()
            cls_loss = F.cross_entropy(logits_f, y)
            adv_preds = model.get_adversary_pred(feat_f, grl_alpha=1.0)
            adv_loss = compute_adversary_loss(adv_preds, sa_labels)
            loss = cls_loss + adv_loss

            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"cls": f"{cls_loss.item():.4f}", "adv": f"{adv_loss.item():.4f}"})

        val_metrics = evaluate_stage1(model, val_loader, device, cfg, cfg.use_amp)
        val_auc = val_metrics["overall_auc"]
        print(
            f"Stage1 Epoch {epoch + 1:03d}: val_loss={val_metrics['val_loss']:.4f}, "
            f"val_auc={val_auc:.4f}  (skipped_nonfinite={skipped})"
        )
        skipped = 0

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_stage1(
    model: DisentangledEncoder,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    use_amp: bool = True,
) -> dict[str, float]:
    del cfg  # unused; kept for signature parity with original
    model.eval()
    logits_all, labels_all = [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with autocast_ctx(use_amp, device):
            logits = model(x)
        logits = logits.float()
        if not torch.isfinite(logits).all():
            continue
        loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        loss_sum += float(loss.item()) * bs
        n += bs
        logits_all.append(logits.cpu())
        labels_all.append(y.cpu())
    if n == 0 or not logits_all:
        return {"val_loss": float("nan"), "overall_auc": float("nan")}
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    probs = F.softmax(logits, dim=1)[:, 1].numpy()
    y_np = labels.numpy()
    try:
        auc = float(roc_auc_score(y_np, probs))
    except Exception:
        auc = float("nan")
    return {"val_loss": loss_sum / max(n, 1), "overall_auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 Training: Refusion (Fusion-style checkpointing: min_improvement vs
# same-seed local image-only ERM). FairREADModel needs SA input, so validation
# uses evaluate_subgroup_report with a custom forward_fn that mirrors the same
# forward path as the original val evaluation (model(x, sa) under autocast).
# ─────────────────────────────────────────────────────────────────────────────

def train_stage2(
    model: FairREADModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    baseline_report: dict[str, float] | None,
) -> tuple[FairREADModel, dict[str, float]]:
    trainable_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("-inf")
    best_report: dict[str, float] = {}
    scaler = make_grad_scaler(cfg.use_amp, device)
    skipped = 0

    def fairread_forward_fn(m: nn.Module, batch: dict, dev: torch.device) -> torch.Tensor:
        x = batch["image"].to(dev, non_blocking=True)
        sa = build_sa_input(batch, cfg.demographic_attrs, dev)
        return m(x, sa)

    for epoch in range(cfg.stage2_epochs):
        model.train()
        model.encoder.eval()
        pbar = tqdm(train_loader, desc=f"Stage2 {epoch + 1}/{cfg.stage2_epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            sa = build_sa_input(batch, cfg.demographic_attrs, device)
            sa_labels = build_sa_labels(batch, cfg.demographic_attrs, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(cfg.use_amp, device):
                logits, adv_preds = model.forward_with_adv(x, sa)
            logits_f = logits.float()
            cls_loss = F.cross_entropy(logits_f, y)
            adv_preds_f = {k: v.float() for k, v in adv_preds.items()}
            adv_loss = compute_adversary_loss(adv_preds_f, sa_labels)
            loss = cls_loss - cfg.adv_alpha * adv_loss

            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"cls": f"{cls_loss.item():.4f}", "adv": f"{adv_loss.item():.4f}"})

        val_report = evaluate_subgroup_report(
            model,
            val_loader,
            device,
            cfg.selection_attr,
            use_amp=cfg.use_amp,
            forward_fn=fairread_forward_fn,
        )
        score = compute_selection_score(
            val_report,
            selection_metric=cfg.selection_metric,
            selection_attr=cfg.selection_attr,
            baseline_report=baseline_report,
        )
        val_report["selection_score"] = float(score) if not math.isnan(score) else float("nan")
        print(
            f"Stage2 Epoch {epoch + 1:03d}: val_loss={val_report['val_loss']:.4f}, "
            f"overall_auc={val_report['overall_auc']:.4f}, {cfg.selection_metric}={score:.4f} "
            f"(skipped_nonfinite={skipped})"
        )
        skipped = 0

        if not math.isnan(score) and score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_report = dict(val_report)

    model.load_state_dict(best_state)
    if not best_report:
        best_report = evaluate_subgroup_report(
            model,
            val_loader,
            device,
            cfg.selection_attr,
            use_amp=cfg.use_amp,
            forward_fn=fairread_forward_fn,
        )
        best_report["selection_score"] = compute_selection_score(
            best_report,
            selection_metric=cfg.selection_metric,
            selection_attr=cfg.selection_attr,
            baseline_report=baseline_report,
        )
    return model, best_report


# ─────────────────────────────────────────────────────────────────────────────
# Full two-stage train (Stage 1 disentanglement → Stage 2 refusion), reused for
# both finetune trials and final multi-seed training.
# ─────────────────────────────────────────────────────────────────────────────

def train_fairread(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    baseline_report: dict[str, float] | None,
    sa_dim: int,
) -> tuple[DisentangledEncoder, FairREADModel, dict[str, float]]:
    disentangled_model = DisentangledEncoder(
        backbone_name=cfg.backbone,
        demo_attrs=cfg.demographic_attrs,
        num_classes=2,
        dropout=cfg.dropout,
    ).to(device)
    disentangled_model = train_stage1(disentangled_model, train_loader, val_loader, device, cfg)

    frozen_encoder = disentangled_model.features
    feat_dim = disentangled_model.feat_dim
    for param in frozen_encoder.parameters():
        param.requires_grad = False

    fairread_model = FairREADModel(
        disentangled_encoder=frozen_encoder,
        feat_dim=feat_dim,
        demo_attrs=cfg.demographic_attrs,
        sa_dim=sa_dim,
        num_classes=2,
        num_cascade_blocks=cfg.num_cascade_blocks,
        mean_std_dim=cfg.mean_std_dim,
        dropout=cfg.dropout,
        conv_reduction=cfg.conv_reduction,
        adv_alpha=cfg.adv_alpha,
    ).to(device)

    fairread_model, best_report = train_stage2(
        fairread_model, train_loader, val_loader, device, cfg, baseline_report
    )
    return disentangled_model, fairread_model, best_report


# ─────────────────────────────────────────────────────────────────────────────
# Test Prediction Collection (needs SA input — not the common collect_test_rows)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_fairread_test_rows(
    model: FairREADModel,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    seed: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        sa = build_sa_input(batch, cfg.demographic_attrs, device)
        with autocast_ctx(cfg.use_amp, device):
            logits = model(x, sa)
        logits = logits.float()
        probs = F.softmax(logits, dim=1)[:, 1]
        pred_cls = (probs >= 0.5).long()
        pos_logit = logits[:, 1]

        sample_ids = batch["sample_id"]
        genders = batch["gender"].tolist()
        races = batch["race"].tolist()
        labels = y.cpu().tolist()
        for i in range(len(sample_ids)):
            rows.append({
                "sample_id": sample_ids[i],
                "true_label": int(labels[i]),
                "logit": float(pos_logit[i].item()),
                "prediction_probability": float(probs[i].item()),
                "pred_class": int(pred_cls[i].item()),
                "gender": int(genders[i]),
                "race": int(races[i]),
                "model_name": model_name,
                "seed": int(seed),
            })
    return rows


def persist_fairread_seed(
    results_root: Path,
    ckpt_root: Path,
    seed: int,
    disentangled_state: Any,
    fairread_state: Any,
    backbone: str,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Atomically save disentangled + fairread checkpoints and test CSV."""
    ckpt_root.mkdir(parents=True, exist_ok=True)
    pending_ckpt = ckpt_root / f"_pending_seed_{seed}"
    final_ckpt = ckpt_root / f"seed_{seed}"
    if pending_ckpt.exists():
        shutil.rmtree(pending_ckpt)
    pending_ckpt.mkdir(parents=True)
    try:
        s1_path = pending_ckpt / f"disentangled_{backbone}.pt"
        s2_path = pending_ckpt / "fairread_refusion.pt"
        torch.save(disentangled_state, s1_path)
        torch.save(fairread_state, s2_path)
        if final_ckpt.exists():
            shutil.rmtree(final_ckpt)
        pending_ckpt.rename(final_ckpt)
        csv_path = save_seed_results(results_root, seed, rows)
        return (
            csv_path,
            final_ckpt / f"disentangled_{backbone}.pt",
            final_ckpt / "fairread_refusion.pt",
        )
    except Exception:
        shutil.rmtree(pending_ckpt, ignore_errors=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# CLI & Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    default_out = str(Path(__file__).resolve().parent / "fairvision_fairread_outputs")
    parser = argparse.ArgumentParser(
        description="FairREAD (Fusion-style finetune) for FairVision Glaucoma."
    )
    add_common_arguments(
        parser,
        default_output_dir=default_out,
        default_backbone=Config.backbone,
        default_epochs=Config.stage1_epochs,
        default_batch_size=Config.batch_size,
    )
    add_finetune_arguments(parser)
    parser.set_defaults(
        lr=Config.lr,
        weight_decay=Config.weight_decay,
        demographic_attrs="gender,race",
    )
    parser.add_argument("--stage1-epochs", type=int, default=Config.stage1_epochs)
    parser.add_argument("--stage2-epochs", type=int, default=Config.stage2_epochs)
    parser.add_argument("--num-cascade-blocks", type=int, default=Config.num_cascade_blocks)
    parser.add_argument("--mean-std-dim", type=int, default=Config.mean_std_dim)
    parser.add_argument("--adv-alpha", type=float, default=Config.adv_alpha)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--conv-reduction", type=int, default=Config.conv_reduction)
    parser.add_argument("--adv-alpha-grid", type=str, default="0.01,0.1,0.5")
    parser.add_argument("--mean-std-dim-grid", type=str, default="128,256,512")
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
    cfg.baseline_selection_attr = resolve_selection_attr(
        args.baseline_selection_attr, ("gender",)
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

    cfg.stage1_epochs = args.stage1_epochs
    cfg.stage2_epochs = args.stage2_epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.num_workers = args.num_workers
    cfg.image_size = args.image_size
    cfg.num_cascade_blocks = args.num_cascade_blocks
    cfg.mean_std_dim = args.mean_std_dim
    cfg.adv_alpha = args.adv_alpha
    cfg.dropout = args.dropout
    cfg.conv_reduction = args.conv_reduction
    cfg.pretrained = not args.no_pretrained
    cfg.use_amp = False if args.no_use_amp else True
    cfg.lr_grid = parse_float_grid(args.lr_grid)
    cfg.weight_decay_grid = parse_float_grid(args.weight_decay_grid)
    cfg.batch_size_grid = parse_int_grid(args.batch_size_grid)
    cfg.adv_alpha_grid = parse_float_grid(args.adv_alpha_grid)
    cfg.mean_std_dim_grid = parse_int_grid(args.mean_std_dim_grid)
    cfg.finetune_summary = args.finetune_summary
    cfg.skip_finetune = bool(args.skip_finetune)
    cfg.finetune_only = bool(args.finetune_only)
    cfg.skip_baseline = bool(args.skip_baseline)
    cfg.baseline_only = bool(args.baseline_only)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Backbone: {cfg.backbone}")
    print(f"AMP: {amp_active(device, cfg.use_amp)}")
    print(f"Seeds: {cfg.seeds}")
    print(f"ERM hparams dir: {cfg.erm_hparams_dir or '(REQUIRED)'}")
    print(f"Selection metric: {cfg.selection_metric}")
    print(f"Baseline selection_attr={cfg.baseline_selection_attr}")
    print(f"Demographic attrs: {cfg.demographic_attrs} ({attr_suffix}) | selection_attr={cfg.selection_attr}")
    print(f"Stage1 epochs: {cfg.stage1_epochs}, Stage2 epochs: {cfg.stage2_epochs}")
    print(f"Cascade blocks: {cfg.num_cascade_blocks}")

    if cfg.baseline_only or needs_local_baseline(cfg.selection_metric):
        ensure_baseline_erm(cfg, cfg.backbone, device)
    if cfg.baseline_only:
        print("baseline_only=1; skipping FairREAD training.")
        return

    run_group = run_group_name("FairREAD", cfg.demographic_attrs, cfg.demographic_subgroup)
    ckpt_root, results_root = get_output_roots(
        cfg.output_dir, "FairREAD", cfg.demographic_attrs, cfg.demographic_subgroup
    )
    print(f"Checkpoint metric: {cfg.selection_metric} (Stage 2)")

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

    sa_dim = compute_sa_dim(cfg.demographic_attrs)
    print(
        f"SA encoding dimension: {sa_dim} "
        f"({', '.join(f'{a}={ATTR_CARDINALITIES[a]}cls' for a in cfg.demographic_attrs)})"
    )

    mid_wd = mid_of(cfg.weight_decay_grid, cfg.weight_decay)
    mid_bs = mid_of(cfg.batch_size_grid, cfg.batch_size)
    mid_adv_alpha = mid_of(cfg.adv_alpha_grid, cfg.adv_alpha)
    mid_mean_std_dim = mid_of(cfg.mean_std_dim_grid, cfg.mean_std_dim)
    initial_hparams: dict[str, Any] = {
        "lr": cfg.lr,
        "weight_decay": float(mid_wd),
        "batch_size": int(mid_bs),
        "adv_alpha": float(mid_adv_alpha),
        "mean_std_dim": int(mid_mean_std_dim),
    }

    def trial_fn(hparams: dict[str, Any], stage: str) -> dict[str, Any]:
        trial_cfg = apply_hparams_to_cfg(cfg, hparams)
        set_seed(int(cfg.seeds[0]))
        setup_cudnn_benchmark(True, device)
        tr, va, _te = _loaders_for(int(trial_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(
            trial_cfg, trial_cfg.backbone, int(cfg.seeds[0]), va, device
        )
        print(f"\n--- Finetune [{stage}] hparams={hparams} ---")
        _dis, _fr, report = train_fairread(trial_cfg, tr, va, device, baseline_report, sa_dim)
        score = float(report.get("selection_score", float("nan")))
        return {"stage": stage, "hparams": dict(hparams), "metrics": report, "selection_score": score}

    stages = [
        ("lr", cfg.lr_grid),
        ("weight_decay", cfg.weight_decay_grid),
        ("batch_size", cfg.batch_size_grid),
        ("adv_alpha", cfg.adv_alpha_grid),
        ("mean_std_dim", cfg.mean_std_dim_grid),
    ]
    grids_meta = {
        "lr": list(cfg.lr_grid),
        "weight_decay": list(cfg.weight_decay_grid),
        "batch_size": list(cfg.batch_size_grid),
        "adv_alpha": list(cfg.adv_alpha_grid),
        "mean_std_dim": list(cfg.mean_std_dim_grid),
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
        print(f"FairREAD | seed={seed} | backbone={train_cfg.backbone} | attrs={train_cfg.demographic_attrs}")
        set_seed(int(seed))
        setup_cudnn_benchmark(True, device)
        tr, va, te = _loaders_for(int(train_cfg.batch_size))
        baseline_report = maybe_load_baseline_report(
            train_cfg, train_cfg.backbone, int(seed), va, device
        )
        disentangled_model, fairread_model, best_report = train_fairread(
            train_cfg, tr, va, device, baseline_report, sa_dim
        )

        run_name = f"{train_cfg.backbone}_FairREAD_{attr_suffix}"
        rows = collect_fairread_test_rows(fairread_model, te, device, run_name, int(seed), train_cfg)

        csv_path, s1_path, s2_path = persist_fairread_seed(
            results_root,
            ckpt_root,
            int(seed),
            disentangled_model.state_dict(),
            fairread_model.state_dict(),
            train_cfg.backbone,
            rows,
        )
        save_json(
            s2_path.parent / f"fairread_{train_cfg.backbone}_val_metrics.json",
            {
                "seed": int(seed),
                "selection_metric": train_cfg.selection_metric,
                "selection_attr": train_cfg.selection_attr,
                "hparams": summary["hparams"],
                "val_metrics": best_report,
            },
        )
        print(f"Saved Stage1 checkpoint: {s1_path}")
        print(f"Saved Stage2 checkpoint: {s2_path}")
        print(f"Saved predictions: {csv_path}")

    print(f"\nFairREAD complete → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
