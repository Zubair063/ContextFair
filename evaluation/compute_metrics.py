#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

GENDER_MAP = {0: "Female", 1: "Male"}  # FairVision
RACE_MAP = {0: "Asian", 1: "Black", 2: "White"}  # FairVision
# CheXpert: sex female=0/male=1; age ≤60→0, >60→1
AGE_MAP = {0: "le60", 1: "gt60"}
# UTKFace encodings (datasets/utkface.py)
UTK_GENDER_MAP = {0: "Male", 1: "Female"}
UTK_RACE_MAP = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
UTK_AGE_MAP = {0: "0_19", 1: "20_39", 2: "40plus"}

CHEXPERT_EVAL_PATHOLOGIES = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)

# Utility defaults (match staged usefulness)
ALPHA = 50.0
LAMBDA_CV = 1.0
EPS = 1e-8

THRESHOLD = 0.5
# CF-AUC / CF-ACC (Eq. 20) penalty scales λ_σ, λ_H
LAMBDA_SIGMA = 5.0
LAMBDA_H = 5.0


# ---------------------------------------------------------------------------
# Core metric helpers (match compute_fairvision_erm_metrics.py)
# ---------------------------------------------------------------------------

def sensitivity_at_specificity(labels: np.ndarray, probs: np.ndarray, target_specificity: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(labels, probs)
    specificity = 1.0 - fpr
    valid = specificity >= target_specificity
    if not np.any(valid):
        return 0.0
    return float(tpr[valid][-1])


def equity_scaled_auc(probs: np.ndarray, labels: np.ndarray, demographics: np.ndarray, alpha: float = 1.0) -> float:
    fpr, tpr, _ = roc_curve(labels, probs)
    overall_auc = float(auc(fpr, tpr))
    group_aucs = []
    for group in np.unique(demographics):
        mask = demographics == group
        g_labels, g_probs = labels[mask], probs[mask]
        if len(np.unique(g_labels)) < 2:
            continue
        fpr_g, tpr_g, _ = roc_curve(g_labels, g_probs)
        group_aucs.append(float(auc(fpr_g, tpr_g)))
    total_deviation = sum(abs(g - overall_auc) for g in group_aucs)
    return overall_auc / (alpha * total_deviation + 1.0)


def equity_scaled_accuracy(
    labels: np.ndarray, probs: np.ndarray, demographics: np.ndarray, alpha: float = 1.0
) -> float:
    y_pred = (probs >= THRESHOLD).astype(int)
    overall_acc = float(np.mean(y_pred == labels))
    group_accs = []
    for group in np.unique(demographics):
        mask = demographics == group
        g_labels, g_preds = labels[mask], y_pred[mask]
        if len(g_labels) == 0:
            continue
        group_accs.append(float(np.mean(g_preds == g_labels)))
    total_deviation = sum(abs(g - overall_acc) for g in group_accs)
    return overall_acc / (alpha * total_deviation + 1.0)


def demographic_parity_difference(probs: np.ndarray, demographics: np.ndarray) -> float:
    y_pred = (probs >= THRESHOLD).astype(int)
    groups = np.unique(demographics)
    rates = {g: float(np.mean(y_pred[demographics == g])) for g in groups}
    return max(
        abs(rates[g1] - rates[g2])
        for i, g1 in enumerate(groups)
        for g2 in list(groups)[i + 1 :]
    )


def equalized_odds(probs: np.ndarray, labels: np.ndarray, demographics: np.ndarray) -> float:
    y_pred = (probs >= THRESHOLD).astype(int)
    groups = np.unique(demographics)
    tpr, fpr = {}, {}
    for g in groups:
        mask = demographics == g
        g_labels, g_preds = labels[mask], y_pred[mask]
        pos, neg = g_labels == 1, g_labels == 0
        tpr[g] = float(np.mean(g_preds[pos])) if int(pos.sum()) > 0 else 0.0
        fpr[g] = float(np.mean(g_preds[neg])) if int(neg.sum()) > 0 else 0.0
    return float(max(max(tpr.values()) - min(tpr.values()), max(fpr.values()) - min(fpr.values())))


def mean_prediction_score_disparity(probs: np.ndarray, demographics: np.ndarray) -> float:
    groups = np.unique(demographics)
    means = [float(np.mean(probs[demographics == g])) for g in groups]
    diffs = [
        abs(means[i] - means[j])
        for i in range(len(means))
        for j in range(i + 1, len(means))
    ]
    return float(np.mean(diffs)) if diffs else float("nan")


def group_auc_dict(probs: np.ndarray, labels: np.ndarray, demographics: np.ndarray, group_map: dict[int, str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for code, name in sorted(group_map.items()):
        mask = demographics == code
        g_labels, g_probs = labels[mask], probs[mask]
        if len(np.unique(g_labels)) < 2:
            result[name] = float("nan")
        else:
            result[name] = float(roc_auc_score(g_labels, g_probs))
    return result


def group_acc_dict(labels: np.ndarray, probs: np.ndarray, demographics: np.ndarray, group_map: dict[int, str]) -> dict[str, float]:
    y_pred = (probs >= THRESHOLD).astype(int)
    result: dict[str, float] = {}
    for code, name in sorted(group_map.items()):
        mask = demographics == code
        g_labels, g_preds = labels[mask], y_pred[mask]
        if len(g_labels) == 0:
            result[name] = float("nan")
        else:
            result[name] = float(np.mean(g_preds == g_labels))
    return result


def _macro_mean(values: list[float]) -> float:
    valid = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return float(np.mean(valid)) if valid else float("nan")


def is_chexpert_df(df: pd.DataFrame) -> bool:
    return "pathology" in df.columns


def detect_dataset(df: pd.DataFrame) -> str:
    """Return 'chexpert' | 'utkface' | 'fairvision' from prediction CSV schema."""
    if is_chexpert_df(df):
        return "chexpert"
    cols = set(df.columns)
    # UTKFace always exports age + gender + race (3 age bins, 5 races).
    if {"age", "gender", "race"}.issubset(cols):
        race_max = int(pd.to_numeric(df["race"], errors="coerce").max())
        age_n = int(pd.to_numeric(df["age"], errors="coerce").nunique())
        if race_max >= 3 or age_n >= 3:
            return "utkface"
    return "fairvision"


def normalize_demographic(demographic: str, dataset: str) -> str:
    """Map aliases and validate demographic vs dataset."""
    d = demographic.lower().strip()
    if d == "sex":
        d = "gender"
    if dataset == "chexpert":
        if d in ("race", "gender_race", "age_race"):
            raise ValueError(
                "CheXpert predictions support --demographic gender|sex|age "
                f"(got {demographic!r})"
            )
        if d not in ("gender", "age"):
            raise ValueError(f"Unsupported CheXpert demographic {demographic!r}")
        return d
    if dataset == "utkface":
        if d not in ("gender", "race", "age", "gender_race", "age_race"):
            raise ValueError(
                "UTKFace supports --demographic gender|race|age|gender_race|age_race "
                f"(got {demographic!r})"
            )
        return d
    # fairvision
    if d not in ("gender", "race", "gender_race"):
        raise ValueError(
            f"FairVision supports --demographic gender|race|gender_race "
            f"(got {demographic!r})"
        )
    return d


def _append_demo_block(
    out: dict[str, float],
    probs: np.ndarray,
    labels: np.ndarray,
    demo: np.ndarray,
    group_map: dict[int, str],
    prefix: str,
) -> None:
    g_auc = group_auc_dict(probs, labels, demo, group_map)
    g_acc = group_acc_dict(labels, probs, demo, group_map)
    for name, val in g_auc.items():
        out[f"{prefix}_Group_AUC_{name}"] = val * 100.0
    for name, val in g_acc.items():
        out[f"{prefix}_Group_ACC_{name}"] = val * 100.0
    out[f"{prefix}_ES_AUC"] = equity_scaled_auc(probs, labels, demo) * 100.0
    out[f"{prefix}_ES_ACC"] = equity_scaled_accuracy(labels, probs, demo) * 100.0
    out[f"{prefix}_DPD"] = demographic_parity_difference(probs, demo) * 100.0
    out[f"{prefix}_DeOdds"] = equalized_odds(probs, labels, demo) * 100.0
    out[f"{prefix}_Mean_PSD"] = mean_prediction_score_disparity(probs, demo) * 100.0


def compute_seed_metrics_binary(
    df: pd.DataFrame, demographic: str, dataset: str = "fairvision"
) -> dict[str, float]:
    """FairVision / UTKFace single-label metrics (values already ×100 where applicable)."""
    probs = df["prediction_probability"].to_numpy(dtype=float)
    labels = df["true_label"].to_numpy(dtype=int)
    y_pred = (probs >= THRESHOLD).astype(int)

    # Multi-class UTKFace targets (age/race as label) may lack both classes for AUC;
    # still report accuracy. Binary gender task is the common case.
    out: dict[str, float] = {
        "Overall_Accuracy": float(np.mean(y_pred == labels)) * 100.0,
        "Se@0.90Sp": sensitivity_at_specificity(labels, probs) * 100.0,
    }
    if len(np.unique(labels)) >= 2:
        out["Overall_AUC"] = float(roc_auc_score(labels, probs)) * 100.0
    else:
        out["Overall_AUC"] = float("nan")

    if dataset == "utkface":
        gender_map, race_map, age_map = UTK_GENDER_MAP, UTK_RACE_MAP, UTK_AGE_MAP
    else:
        gender_map, race_map, age_map = GENDER_MAP, RACE_MAP, AGE_MAP

    if demographic in ("gender", "gender_race"):
        _append_demo_block(
            out, probs, labels, df["gender"].to_numpy(dtype=int), gender_map, "Gender"
        )

    if demographic in ("race", "gender_race", "age_race"):
        _append_demo_block(
            out, probs, labels, df["race"].to_numpy(dtype=int), race_map, "Race"
        )

    if demographic in ("age", "age_race") and dataset == "utkface":
        if "age" not in df.columns:
            raise KeyError("UTKFace predictions missing 'age' column")
        _append_demo_block(
            out, probs, labels, df["age"].to_numpy(dtype=int), age_map, "Age"
        )
    return out


def _chexpert_pathology_rows(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    present = [p for p in CHEXPERT_EVAL_PATHOLOGIES if (df["pathology"] == p).any()]
    if not present:
        raise ValueError(
            "CheXpert CSV has a pathology column but none of the 5 competition "
            f"findings: {CHEXPERT_EVAL_PATHOLOGIES}"
        )
    return [(p, df[df["pathology"] == p]) for p in present]


def _chexpert_demo_col(demographic: str) -> tuple[str, dict[int, str], str]:
    """Return (dataframe column, group_map, metric prefix)."""
    if demographic == "gender":
        return "sex", GENDER_MAP, "Gender"
    if demographic == "age":
        return "age", AGE_MAP, "Age"
    raise ValueError(f"CheXpert demographic must be gender|age; got {demographic!r}")


def compute_seed_metrics_chexpert(df: pd.DataFrame, demographic: str) -> dict[str, float]:
    """Macro-average metrics over the 5 CheXpert competition pathologies."""
    demo_col, group_map, prefix = _chexpert_demo_col(demographic)
    if demo_col not in df.columns:
        raise KeyError(
            f"CheXpert predictions missing '{demo_col}' column "
            f"(needed for --demographic {demographic})"
        )

    overall_aucs: list[float] = []
    overall_accs: list[float] = []
    overall_sens: list[float] = []
    per_path_subgroup: list[dict[str, float]] = []

    for _pathology, sub in _chexpert_pathology_rows(df):
        probs = sub["prediction_probability"].to_numpy(dtype=float)
        labels = sub["true_label"].to_numpy(dtype=int)
        demo = sub[demo_col].to_numpy(dtype=int)
        y_pred = (probs >= THRESHOLD).astype(int)

        overall_accs.append(float(np.mean(y_pred == labels)))
        overall_sens.append(sensitivity_at_specificity(labels, probs))
        if len(np.unique(labels)) >= 2:
            overall_aucs.append(float(roc_auc_score(labels, probs)))
        else:
            overall_aucs.append(float("nan"))

        g_auc = group_auc_dict(probs, labels, demo, group_map)
        g_acc = group_acc_dict(labels, probs, demo, group_map)
        row: dict[str, float] = {
            f"{prefix}_ES_AUC": equity_scaled_auc(probs, labels, demo),
            f"{prefix}_ES_ACC": equity_scaled_accuracy(labels, probs, demo),
            f"{prefix}_DPD": demographic_parity_difference(probs, demo),
            f"{prefix}_DeOdds": equalized_odds(probs, labels, demo),
            f"{prefix}_Mean_PSD": mean_prediction_score_disparity(probs, demo),
        }
        for name, val in g_auc.items():
            row[f"{prefix}_Group_AUC_{name}"] = val
        for name, val in g_acc.items():
            row[f"{prefix}_Group_ACC_{name}"] = val
        per_path_subgroup.append(row)

    out: dict[str, float] = {
        "Overall_Accuracy": _macro_mean(overall_accs) * 100.0,
        "Overall_AUC": _macro_mean(overall_aucs) * 100.0,
        "Se@0.90Sp": _macro_mean(overall_sens) * 100.0,
    }
    keys = sorted({k for row in per_path_subgroup for k in row})
    for key in keys:
        out[key] = _macro_mean([row[key] for row in per_path_subgroup]) * 100.0
    return out


def compute_seed_metrics(
    df: pd.DataFrame, demographic: str, dataset: str | None = None
) -> dict[str, float]:
    """Dispatch FairVision / UTKFace binary vs CheXpert multilabel macro metrics."""
    if dataset is None:
        dataset = detect_dataset(df)
    demo = normalize_demographic(demographic, dataset=dataset)
    if dataset == "chexpert":
        return compute_seed_metrics_chexpert(df, demo)
    return compute_seed_metrics_binary(df, demo, dataset=dataset)


# ---------------------------------------------------------------------------
# Utility + CF-AUC (same-seed baseline)
# ---------------------------------------------------------------------------

def sigmoid_np(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def compute_cf_metric(overall: float, improvements: list[float]) -> tuple[float, float]:
    """Contextual Fairness CF-M (Eq. 20); M is Overall AUC or Accuracy.

    Inputs use percentage scale (×100). With Δ_g on that scale, Eq. (20)'s
    factors ``1 + Δ̄`` become ``100 + Δ̄`` (equivalent to fractional [0,1] form).

    Eq. (19) — mean / std of subgroup gains ``improvements`` = {Δ_g}:
        Δ̄  = (1/|G|) Σ_g Δ_g
        Δ_σ = sqrt( (1/|G|) Σ_g (Δ_g - Δ̄)^2 )

    Subgroup harm H (Eq. 6): H = |Σ_{g: Δ_g < 0} Δ_g|

    Eq. (20) — Contextual Fairness:
        CF-M = M · (1 + Δ̄) / (1 + Δ̄ + λ_σ Δ_σ + λ_H H)
             ≈ M · (100 + Δ̄) / (100 + Δ̄ + λ_σ Δ_σ + λ_H H)   [%-scale]

    Returns ``(Δ̄, CF-M)``. λ_σ = LAMBDA_SIGMA, λ_H = LAMBDA_H.
    """
    if not improvements:
        return float("nan"), float("nan")

    # Eq. (19): Δ̄ and Δ_σ over subgroup gains Δ_g
    mu_delta = float(np.mean(improvements))  # Δ̄
    # Population std (1/|G|), matching Eq. (19)
    sigma_delta = float(np.std(improvements, ddof=0)) if len(improvements) >= 1 else 0.0  # Δ_σ

    # Eq. (6): subgroup harm H
    abs_delta_h = abs(float(sum(x for x in improvements if x < 0)))  # H

    # Eq. (20): CF-M  (percentage-scale form of 1+Δ̄ → 100+Δ̄)
    denom_base = 100.0 + mu_delta
    denom = denom_base + LAMBDA_SIGMA * sigma_delta + LAMBDA_H * abs_delta_h
    if denom == 0.0:
        return mu_delta, float("nan")
    cf = overall * denom_base / denom
    return mu_delta, float(cf)


def add_cf_metrics(
    metrics: dict[str, float],
    base_metrics: dict[str, float],
    demographic: str,
    prefix: str | None = None,
    dataset: str = "fairvision",
) -> None:
    """Attach CF-AUC / CF-ACC (Eq. 20) vs attribute-agnostic baseline.

    For each subgroup g, Δ_g (Eq. 5) = M_g(proposed) − M_g(baseline), then
    CF-M via ``compute_cf_metric`` (Eq. 19–20).
    """
    if prefix is not None:
        tag = prefix
    elif demographic == "gender":
        tag = "Gender"
    elif demographic == "race":
        tag = "Race"
    elif demographic == "age":
        tag = "Age"
    else:
        tag = demographic
    for kind, overall_key in (("AUC", "Overall_AUC"), ("ACC", "Overall_Accuracy")):
        names = subgroup_metric_names(demographic, kind, dataset=dataset)
        # Eq. (5): subgroup gains Δ_g
        improvements = [float(metrics[n]) - float(base_metrics[n]) for n in names]
        # Eq. (19)–(20): Δ̄, Δ_σ, H → CF-AUC or CF-ACC
        mean_imp, cf = compute_cf_metric(float(metrics[overall_key]), improvements)
        metrics[f"{tag}_Mean_{kind}_improvement"] = mean_imp  # Δ̄
        metrics[f"{tag}_CF_{kind}"] = cf  # CF-M


def compute_utility_score(
    overall_base: float,
    overall_prop: float,
    subgroup_base: list[float],
    subgroup_prop: list[float],
    alpha: float = ALPHA,
    lambda_cv: float = LAMBDA_CV,
    eps: float = EPS,
) -> tuple[float, dict[str, Any]]:
    """
    Usefulness / utility score. Inputs on percentage scale; converted to [0,1].

    u = 1[B>H] · σ(α·d) · exp(-λ_CV·CV)
    """
    scale = 0.01
    auc_b = overall_base * scale
    auc_d = overall_prop * scale
    d = auc_d - auc_b
    raw_deltas = [(p - b) * scale for p, b in zip(subgroup_prop, subgroup_base)]

    improvement_gap = float(max(raw_deltas) - min(raw_deltas)) if len(raw_deltas) > 1 else float("nan")
    if len(raw_deltas) == 0 or math.isnan(d):
        return 0.0, {"d": d, "gate_passed": 0.0, "subgroup_improvement_gap": improvement_gap}

    delta = np.asarray(raw_deltas, dtype=np.float64)
    delta_pos = np.maximum(0.0, delta)
    delta_neg = np.maximum(0.0, -delta)
    B, H = float(delta_pos.sum()), float(delta_neg.sum())
    if B <= H:
        return 0.0, {"d": d, "B": B, "H": H, "gate_passed": 0.0, "subgroup_improvement_gap": improvement_gap}

    mu = float(delta_pos.mean())
    sigma = float(delta_pos.std())  # population std
    if mu <= eps:
        return 0.0, {"d": d, "B": B, "H": H, "gate_passed": 0.0, "subgroup_improvement_gap": improvement_gap}

    CV = sigma / (mu + eps)
    penalty = math.exp(-lambda_cv * CV)
    u = sigmoid_np(alpha * d) * penalty
    return float(u), {
        "d": d,
        "B": B,
        "H": H,
        "CV": CV,
        "penalty": penalty,
        "gate_passed": 1.0,
        "subgroup_improvement_gap": improvement_gap,
        "deltas": raw_deltas,
    }


def subgroup_metric_names(demographic: str, kind: str, dataset: str = "fairvision") -> list[str]:
    """kind: 'AUC' or 'ACC'."""
    if demographic == "gender":
        return [f"Gender_Group_{kind}_Female", f"Gender_Group_{kind}_Male"]
    if demographic == "race":
        if dataset == "utkface":
            return [
                f"Race_Group_{kind}_White",
                f"Race_Group_{kind}_Black",
                f"Race_Group_{kind}_Asian",
                f"Race_Group_{kind}_Indian",
                f"Race_Group_{kind}_Others",
            ]
        return [
            f"Race_Group_{kind}_Asian",
            f"Race_Group_{kind}_Black",
            f"Race_Group_{kind}_White",
        ]
    if demographic == "age":
        if dataset == "utkface":
            return [
                f"Age_Group_{kind}_0_19",
                f"Age_Group_{kind}_20_39",
                f"Age_Group_{kind}_40plus",
            ]
        return [f"Age_Group_{kind}_le60", f"Age_Group_{kind}_gt60"]
    raise ValueError(f"Utility requires demographic gender|race|age; got {demographic!r}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def discover_seed_csvs(results_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for p in sorted(results_dir.glob("seed_*")):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"seed_(\d+)", p.name)
        if not m:
            continue
        csv_path = p / "test_predictions.csv"
        if csv_path.is_file():
            found.append((int(m.group(1)), csv_path))
    return sorted(found, key=lambda x: x[0])


def model_names_in_csv(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["model_name"])
    return sorted(df["model_name"].astype(str).unique().tolist())


def short_model_name(model_name: str) -> str:
    """Map a full model_name to its backbone stem for baseline pairing.

    Examples:
      vgg16_image_only → vgg16
      vgg16_proposed_gender_fusion_film → vgg16
      resnet50_naive_demo → resnet50
    """
    name = str(model_name)
    for marker in ("_image_only", "_naive_demo", "_proposed"):
        if marker in name:
            return name.split(marker, 1)[0]
    return name


def build_transposed_table(seed_metrics: dict[str, dict[str, float]], metric_order: list[str]) -> pd.DataFrame:
    seed_cols = sorted(seed_metrics.keys(), key=lambda s: int(s.replace("seed_", "")))
    rows = []
    for metric in metric_order:
        if not any(metric in seed_metrics[s] for s in seed_cols):
            continue
        row: dict[str, Any] = {"Metric": metric}
        values = []
        for seed_col in seed_cols:
            val = seed_metrics[seed_col].get(metric, float("nan"))
            if isinstance(val, float) and not math.isnan(val):
                # Utility stays in [0,1]; others already ×100
                if metric.startswith("Utility"):
                    rounded = round(val, 6)
                elif "CF_" in metric or metric.endswith("_improvement"):
                    rounded = round(val, 4)
                else:
                    rounded = round(val, 2)
            else:
                rounded = val
            row[seed_col] = rounded
            if isinstance(val, (int, float)) and not math.isnan(float(val)):
                values.append(float(val))
        if values:
            mean_v = float(np.mean(values))
            std_v = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if metric.startswith("Utility"):
                row["Mean"] = round(mean_v, 6)
                row["Std"] = round(std_v, 6)
                row["Mean ± Std"] = f"{mean_v:.4f} ± {std_v:.4f}"
            elif "CF_" in metric or metric.endswith("_improvement"):
                row["Mean"] = round(mean_v, 4)
                row["Std"] = round(std_v, 4)
                row["Mean ± Std"] = f"{mean_v:.4f} ± {std_v:.4f}"
            else:
                row["Mean"] = round(mean_v, 2)
                row["Std"] = round(std_v, 2)
                row["Mean ± Std"] = f"{mean_v:.2f} ± {std_v:.2f}"
        else:
            row["Mean"] = float("nan")
            row["Std"] = float("nan")
            row["Mean ± Std"] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def metric_order_for(demographic: str, with_baseline: bool, dataset: str = "fairvision") -> list[str]:
    order = ["Overall_AUC", "Overall_Accuracy", "Se@0.90Sp"]
    if demographic in ("gender", "gender_race"):
        order += [
            "Gender_Group_AUC_Female",
            "Gender_Group_AUC_Male",
            "Gender_Group_ACC_Female",
            "Gender_Group_ACC_Male",
            "Gender_ES_AUC",
            "Gender_ES_ACC",
            "Gender_DPD",
            "Gender_DeOdds",
            "Gender_Mean_PSD",
        ]
        if with_baseline:
            order += [
                "Gender_Mean_AUC_improvement",
                "Gender_CF_AUC",
                "Gender_Mean_ACC_improvement",
                "Gender_CF_ACC",
            ]
    if demographic in ("race", "gender_race", "age_race"):
        if dataset == "utkface":
            order += [
                "Race_Group_AUC_White",
                "Race_Group_AUC_Black",
                "Race_Group_AUC_Asian",
                "Race_Group_AUC_Indian",
                "Race_Group_AUC_Others",
                "Race_Group_ACC_White",
                "Race_Group_ACC_Black",
                "Race_Group_ACC_Asian",
                "Race_Group_ACC_Indian",
                "Race_Group_ACC_Others",
            ]
        else:
            order += [
                "Race_Group_AUC_Asian",
                "Race_Group_AUC_Black",
                "Race_Group_AUC_White",
                "Race_Group_ACC_Asian",
                "Race_Group_ACC_Black",
                "Race_Group_ACC_White",
            ]
        order += [
            "Race_ES_AUC",
            "Race_ES_ACC",
            "Race_DPD",
            "Race_DeOdds",
            "Race_Mean_PSD",
        ]
        if with_baseline:
            order += [
                "Race_Mean_AUC_improvement",
                "Race_CF_AUC",
                "Race_Mean_ACC_improvement",
                "Race_CF_ACC",
            ]
    if demographic in ("age", "age_race"):
        if dataset == "utkface":
            order += [
                "Age_Group_AUC_0_19",
                "Age_Group_AUC_20_39",
                "Age_Group_AUC_40plus",
                "Age_Group_ACC_0_19",
                "Age_Group_ACC_20_39",
                "Age_Group_ACC_40plus",
            ]
        else:
            order += [
                "Age_Group_AUC_le60",
                "Age_Group_AUC_gt60",
                "Age_Group_ACC_le60",
                "Age_Group_ACC_gt60",
            ]
        order += [
            "Age_ES_AUC",
            "Age_ES_ACC",
            "Age_DPD",
            "Age_DeOdds",
            "Age_Mean_PSD",
        ]
        if with_baseline:
            order += [
                "Age_Mean_AUC_improvement",
                "Age_CF_AUC",
                "Age_Mean_ACC_improvement",
                "Age_CF_ACC",
            ]
    if with_baseline and demographic in ("gender", "race", "age"):
        order += ["Utility_Score", "Utility_Score_ACC"]
    return order


def evaluate_run(
    predictions_dir: Path,
    demographic: str,
    baseline_dir: Path | None,
    output_dir: Path,
    alpha: float,
    lambda_cv: float,
) -> None:
    pred_seeds = discover_seed_csvs(predictions_dir)
    if not pred_seeds:
        raise FileNotFoundError(f"No seed_*/test_predictions.csv under {predictions_dir}")

    baseline_by_seed: dict[int, Path] = {}
    if baseline_dir is not None:
        baseline_by_seed = {s: p for s, p in discover_seed_csvs(baseline_dir)}
        if not baseline_by_seed:
            raise FileNotFoundError(f"No baseline CSVs under {baseline_dir}")

    # Need enough rows for UTKFace race/age cardinality detection.
    sample_probe = pd.read_csv(pred_seeds[0][1], nrows=200)
    dataset = detect_dataset(sample_probe)
    demographic = normalize_demographic(demographic, dataset=dataset)
    if dataset == "chexpert":
        print(
            "Detected CheXpert predictions: metrics are macro-averaged over "
            f"{list(CHEXPERT_EVAL_PATHOLOGIES)}"
        )
    elif dataset == "utkface":
        print("Detected UTKFace predictions (age 3-bin / race 5-group encodings).")
    else:
        print("Detected FairVision-style binary predictions.")

    # Pair by matching seed id when possible; else by sorted rank.
    if baseline_by_seed:
        common = sorted(set(s for s, _ in pred_seeds) & set(baseline_by_seed))
        if common:
            pairs = [(s, s) for s in common]  # (prop_seed, base_seed)
            print(f"Paired {len(pairs)} seeds by matching seed id.")
        else:
            b_sorted = sorted(baseline_by_seed)
            p_sorted = [s for s, _ in pred_seeds]
            n = min(len(b_sorted), len(p_sorted))
            pairs = [(p_sorted[i], b_sorted[i]) for i in range(n)]
            print(f"WARNING: no overlapping seed ids; paired by rank ({n} pairs).")
    else:
        pairs = [(s, None) for s, _ in pred_seeds]
        print("No baseline: skipping Utility Score.")

    pred_csv_map = {s: p for s, p in pred_seeds}
    sample_csv = pred_csv_map[pairs[0][0]]
    models = model_names_in_csv(sample_csv)
    print(f"Models: {models}")
    print(f"Demographic mode: {demographic}")

    output_dir.mkdir(parents=True, exist_ok=True)
    util_demo = demographic if demographic in ("gender", "race", "age") else None
    if demographic == "gender_race" and baseline_dir is not None:
        print("NOTE: utility requires a single demographic; computing for gender and race separately.")
    if demographic == "age_race" and baseline_dir is not None:
        print("NOTE: utility requires a single demographic; computing for age and race separately.")

    def _pick_baseline_df(bdf: pd.DataFrame, model_name: str) -> pd.DataFrame:
        base_models = bdf["model_name"].astype(str).unique().tolist()
        prefer = short_model_name(model_name) + "_image_only"
        if prefer in base_models:
            return bdf[bdf["model_name"].astype(str) == prefer]
        img_only = [m for m in base_models if m.endswith("_image_only")]
        pick = img_only[0] if img_only else base_models[0]
        return bdf[bdf["model_name"].astype(str) == pick]

    def _attach_utility(
        metrics: dict[str, float],
        base_metrics: dict[str, float],
        demo: str,
    ) -> tuple[float, float, dict[str, Any], dict[str, Any]]:
        add_cf_metrics(metrics, base_metrics, demo, dataset=dataset)
        u_auc, meta_auc = compute_utility_score(
            base_metrics["Overall_AUC"],
            metrics["Overall_AUC"],
            [base_metrics[n] for n in subgroup_metric_names(demo, "AUC", dataset=dataset)],
            [metrics[n] for n in subgroup_metric_names(demo, "AUC", dataset=dataset)],
            alpha=alpha,
            lambda_cv=lambda_cv,
        )
        u_acc, meta_acc = compute_utility_score(
            base_metrics["Overall_Accuracy"],
            metrics["Overall_Accuracy"],
            [base_metrics[n] for n in subgroup_metric_names(demo, "ACC", dataset=dataset)],
            [metrics[n] for n in subgroup_metric_names(demo, "ACC", dataset=dataset)],
            alpha=alpha,
            lambda_cv=lambda_cv,
        )
        return u_auc, u_acc, meta_auc, meta_acc

    for model_name in models:
        seed_metrics: dict[str, dict[str, float]] = {}
        pairwise_rows: list[dict[str, Any]] = []

        for prop_seed, base_seed in pairs:
            df = pd.read_csv(pred_csv_map[prop_seed])
            df = df[df["model_name"].astype(str) == model_name].copy()
            if df.empty:
                print(f"WARNING: no rows for model={model_name} seed={prop_seed}")
                continue
            metrics = compute_seed_metrics(df, demographic, dataset=dataset)

            if base_seed is not None and util_demo in ("gender", "race", "age"):
                bdf = _pick_baseline_df(pd.read_csv(baseline_by_seed[base_seed]), model_name)
                base_metrics = compute_seed_metrics(bdf, util_demo, dataset=dataset)
                u_auc, u_acc, meta_auc, meta_acc = _attach_utility(metrics, base_metrics, util_demo)
                metrics["Utility_Score"] = u_auc
                metrics["Utility_Score_ACC"] = u_acc

                tag = {"gender": "Gender", "race": "Race", "age": "Age"}[util_demo]
                pairwise_rows.append(
                    {
                        "model_name": model_name,
                        "proposed_seed": prop_seed,
                        "baseline_seed": base_seed,
                        "Overall_AUC": round(metrics["Overall_AUC"], 2),
                        "Overall_Accuracy": round(metrics["Overall_Accuracy"], 2),
                        f"{tag}_ES_AUC": round(metrics[f"{tag}_ES_AUC"], 2),
                        f"{tag}_DPD": round(metrics[f"{tag}_DPD"], 2),
                        f"{tag}_DeOdds": round(metrics[f"{tag}_DeOdds"], 2),
                        f"{tag}_CF_AUC": round(metrics[f"{tag}_CF_AUC"], 4),
                        "Utility_Score": round(u_auc, 6),
                        "Utility_Score_ACC": round(u_acc, 6),
                        "utility_gate_passed": meta_auc.get("gate_passed", 0.0),
                        "d_auc": meta_auc.get("d", float("nan")),
                        "d_acc": meta_acc.get("d", float("nan")),
                    }
                )

            elif base_seed is not None and demographic in ("gender_race", "age_race"):
                bdf = _pick_baseline_df(pd.read_csv(baseline_by_seed[base_seed]), model_name)
                demos = ("gender", "race") if demographic == "gender_race" else ("age", "race")
                for demo in demos:
                    base_metrics = compute_seed_metrics(bdf, demo, dataset=dataset)
                    u_auc, _, _, _ = _attach_utility(metrics, base_metrics, demo)
                    metrics[f"Utility_Score_{demo}"] = u_auc

            seed_metrics[f"seed_{prop_seed}"] = metrics

        with_baseline = baseline_dir is not None
        order = metric_order_for(demographic, with_baseline=with_baseline, dataset=dataset)
        if demographic == "gender_race" and with_baseline:
            order += ["Utility_Score_gender", "Utility_Score_race"]
        if demographic == "age_race" and with_baseline:
            order += ["Utility_Score_age", "Utility_Score_race"]

        table = build_transposed_table(seed_metrics, order)
        short = short_model_name(model_name)
        out_csv = output_dir / f"{short}_metrics.csv"
        table.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

        dataset_label = {
            "chexpert": "CheXpert (macro over 5 pathologies)",
            "utkface": "UTKFace",
            "fairvision": "FairVision",
        }.get(dataset, dataset)
        summary_path = output_dir / f"{short}_metrics_summary.txt"
        lines = [
            "=" * 88,
            f"Metrics summary — {model_name}",
            f"predictions: {predictions_dir}",
            f"baseline:    {baseline_dir}",
            f"demographic: {demographic}",
            f"dataset:     {dataset_label}",
            "=" * 88,
            "",
        ]
        if not table.empty and "Mean ± Std" in table.columns:
            for _, row in table.iterrows():
                lines.append(f"{row['Metric']:<32} {row['Mean ± Std']}")
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {summary_path}")

        if pairwise_rows:
            pair_path = output_dir / f"{short}_utility_pairwise.csv"
            pd.DataFrame(pairwise_rows).to_csv(pair_path, index=False)
            print(f"Wrote {pair_path}")

    meta = {
        "predictions_dir": str(predictions_dir),
        "baseline_dir": str(baseline_dir) if baseline_dir else None,
        "demographic": demographic,
        "dataset": dataset,
        "chexpert_macro": dataset == "chexpert",
        "chexpert_pathologies": list(CHEXPERT_EVAL_PATHOLOGIES) if dataset == "chexpert" else None,
        "compute_cf_auc": baseline_dir is not None,
        "utility_alpha": alpha,
        "utility_lambda_cv": lambda_cv,
        "n_seed_pairs": len(pairs),
    }
    (output_dir / "eval_config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute FairVision / UTKFace / CheXpert metrics from test_predictions.csv"
    )
    p.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Dir with seed_*/test_predictions.csv (proposed / ERM+demo / etc.)",
    )
    p.add_argument(
        "--baseline-dir",
        type=str,
        default="",
        help="Optional image-only ERM results dir (for Utility Score; same-seed pairing).",
    )
    p.add_argument(
        "--demographic",
        type=str,
        choices=["gender", "sex", "race", "gender_race", "age", "age_race"],
        default="gender",
        help=(
            "Demographic fairness metrics / utility pairing. "
            "FairVision: gender|race|gender_race. "
            "UTKFace: gender|race|age|gender_race|age_race. "
            "CheXpert (auto-detected): gender|sex|age."
        ),
    )
    p.add_argument("--output-dir", type=str, default="", help="Default: <predictions-dir>/metrics")
    p.add_argument("--alpha", type=float, default=ALPHA, help="Utility sigmoid scale α")
    p.add_argument("--lambda-cv", type=float, default=LAMBDA_CV, help="Utility CV penalty λ_CV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictions_dir = Path(args.predictions_dir).resolve()
    baseline_dir = Path(args.baseline_dir).resolve() if str(args.baseline_dir).strip() else None
    output_dir = (
        Path(args.output_dir).resolve()
        if str(args.output_dir).strip()
        else predictions_dir / "metrics"
    )
    evaluate_run(
        predictions_dir=predictions_dir,
        demographic=args.demographic,
        baseline_dir=baseline_dir,
        output_dir=output_dir,
        alpha=float(args.alpha),
        lambda_cv=float(args.lambda_cv),
    )


if __name__ == "__main__":
    main()
