from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

FINDING_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
]
NUM_CLASSES = len(FINDING_COLS)

EVAL_FINDING_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
EVAL_CLASS_INDICES: tuple[int, ...] = tuple(FINDING_COLS.index(c) for c in EVAL_FINDING_COLS)
NUM_EVAL_CLASSES = len(EVAL_FINDING_COLS)

SEX_MAP = {"male": 1, "female": 0, "Male": 1, "Female": 0}
SEX_SUBGROUP_NAMES = ("female", "male")
AGE_SUBGROUP_NAMES = ("age_le_60", "age_gt_60")
AGE_GROUP_THRESHOLD = 60  # 0: <=60, 1: >60

DEFAULT_DATA_ROOT = "./data/CheXpert"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRAIN_DISEASE_CHOICES = (5, 13)
DEFAULT_TRAIN_DISEASES = 13

AugStyle = Literal["erm", "contextfair", "none"]


def age_to_group(age: float | int, threshold: int = AGE_GROUP_THRESHOLD) -> int:
    """Map CheXpert age to binary group: 0 = <=threshold, 1 = >threshold."""
    try:
        age_val = float(age)
    except (TypeError, ValueError):
        return 0
    if age_val <= 0 or np.isnan(age_val):
        return 0
    return 0 if age_val <= threshold else 1


def normalize_labels(labels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Map uncertain CheXpert labels (-1) to 0 for BCE training."""
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels.astype(np.float32))
    labels = labels.float()
    labels[labels == -1] = 0.0
    return labels


def slice_eval_labels(labels: torch.Tensor) -> torch.Tensor:
    """Select the 5 competition disease columns from a 13-d tensor."""
    idx = torch.tensor(EVAL_CLASS_INDICES, device=labels.device, dtype=torch.long)
    return labels.index_select(-1, idx)


def resolve_train_diseases(train_diseases: int | str) -> int:
    n = int(train_diseases)
    if n not in TRAIN_DISEASE_CHOICES:
        raise ValueError(f"train_diseases must be one of {TRAIN_DISEASE_CHOICES}, got {train_diseases!r}")
    return n


def num_classes_for_train_diseases(train_diseases: int | str) -> int:
    n = resolve_train_diseases(train_diseases)
    return NUM_EVAL_CLASSES if n == 5 else NUM_CLASSES


def extract_patient_study(path: str) -> tuple[str | None, str | None]:
    match = re.search(r"patient(\d+)/study(\d+)", path)
    if match:
        return match.group(1), match.group(2)
    return None, None


def _build_transforms(image_size: int, train: bool, aug_style: AugStyle | str) -> T.Compose:
    style = str(aug_style).strip().lower()
    if style not in ("erm", "contextfair", "none"):
        raise ValueError(f"aug_style must be erm|contextfair|none; got {aug_style!r}")

    resize = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR)
    normalize = T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))

    if not train or style == "none":
        return T.Compose([resize, T.ToTensor(), normalize])

    if style == "contextfair":
        # Staged / ContextFair CheXpert: flip only (no rotation / jitter).
        return T.Compose(
            [
                resize,
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalize,
            ]
        )

    # ERM / Fusion / Fairness default.
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            normalize,
        ]
    )


class CheXpertFrontalDataset(Dataset):
    """One sample per frontal image row in the CSV (train / valid / test)."""

    def __init__(
        self,
        data_root: str,
        csv_path: str,
        split_name: str,
        image_size: int = 224,
        train: bool = False,
        aug_style: AugStyle | str = "erm",
    ):
        self.data_root = os.path.abspath(data_root)
        self.split_name = split_name
        self.train = train
        self.image_size = image_size
        self.aug_style = aug_style

        df = pd.read_csv(csv_path)
        if "Frontal/Lateral" in df.columns:
            view = df["Frontal/Lateral"].astype(str).str.strip().str.lower()
            df = df[view == "frontal"].reset_index(drop=True)
        df["rel_path"] = df["Path"].astype(str).str.replace("CheXpert-v1.0-small/", "", regex=False)

        self.rel_paths = df["rel_path"].tolist()
        self.labels = df[FINDING_COLS].astype(np.float32).values
        sex_series = df["Sex"].astype(str).str.strip().str.lower()
        self.sex = sex_series.map({"male": 1, "female": 0}).fillna(0).astype(np.int64).values
        if "Age" in df.columns:
            self.age = np.array(
                [age_to_group(a) for a in df["Age"].astype(float).values],
                dtype=np.int64,
            )
        else:
            self.age = np.zeros(len(df), dtype=np.int64)
        self.sample_ids = [
            Path(p).stem if (p := str(rel)) else f"{split_name}_{i}"
            for i, rel in enumerate(self.rel_paths)
        ]
        self.transform = _build_transforms(image_size, train=train, aug_style=aug_style)

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        full_path = os.path.join(self.data_root, self.rel_paths[idx])
        img = Image.open(full_path).convert("RGB")
        image = self.transform(img)
        labels = normalize_labels(self.labels[idx])
        return {
            "image": image,
            "labels": labels,
            "sex": torch.tensor(self.sex[idx], dtype=torch.long),
            "age": torch.tensor(self.age[idx], dtype=torch.long),
            "sample_id": self.sample_ids[idx],
            "rel_path": self.rel_paths[idx],
        }


class CheXpertMultimodalDataset(Dataset):
    """Patient–study samples with frontal image and optional lateral (zeros if missing)."""

    def __init__(
        self,
        data_root: str,
        csv_path: str,
        split_name: str,
        image_size: int = 224,
        train: bool = False,
        aug_style: AugStyle | str = "erm",
    ):
        self.data_root = os.path.abspath(data_root)
        self.split_name = split_name
        self.train = train
        self.image_size = image_size

        df = pd.read_csv(csv_path)
        df["rel_path"] = df["Path"].astype(str).str.replace("CheXpert-v1.0-small/", "", regex=False)

        groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"frontal": None, "lateral": None, "row": None}
        )
        for _, row in df.iterrows():
            patient_id, study_id = extract_patient_study(row["Path"])
            if patient_id is None or study_id is None:
                continue
            key = (patient_id, study_id)
            modality = str(row["Frontal/Lateral"]).strip().lower()
            if modality == "frontal":
                if groups[key]["frontal"] is None:
                    groups[key]["frontal"] = row
                    groups[key]["row"] = row
            elif modality == "lateral":
                if groups[key]["lateral"] is None:
                    groups[key]["lateral"] = row

        self.samples: list[dict[str, Any]] = []
        for (patient_id, study_id), data in groups.items():
            frontal_row = data["frontal"]
            if frontal_row is None:
                continue
            lateral_row = data["lateral"]
            main_row = data["row"]
            sex_str = str(main_row["Sex"]).strip().lower()
            self.samples.append(
                {
                    "sample_id": f"patient{patient_id}_study{study_id}",
                    "frontal_path": str(frontal_row["rel_path"]),
                    "lateral_path": str(lateral_row["rel_path"]) if lateral_row is not None else None,
                    "labels": main_row[FINDING_COLS].astype(np.float32).values,
                    "sex": int(SEX_MAP.get(sex_str, 0)),
                }
            )

        self.transform = _build_transforms(image_size, train=train, aug_style=aug_style)
        self.norm_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.norm_std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    @property
    def has_lateral_list(self) -> list[int]:
        return [1 if s["lateral_path"] is not None else 0 for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        full_path = os.path.join(self.data_root, rel_path)
        img = Image.open(full_path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        frontal = self._load_image(sample["frontal_path"])
        if sample["lateral_path"] is not None:
            lateral = self._load_image(sample["lateral_path"])
            has_lateral = torch.tensor(1.0, dtype=torch.float32)
        else:
            lateral = torch.zeros(3, self.image_size, self.image_size)
            lateral = (lateral - self.norm_mean) / self.norm_std
            has_lateral = torch.tensor(0.0, dtype=torch.float32)

        labels = normalize_labels(sample["labels"])
        return {
            "frontal": frontal,
            "lateral": lateral,
            "labels": labels,
            "sex": torch.tensor(sample["sex"], dtype=torch.long),
            "has_lateral": has_lateral,
            "sample_id": sample["sample_id"],
        }


class ModalityBalancedBatchSampler(Sampler):
    """Batch sampler balancing lateral-present vs frontal-only samples."""

    def __init__(
        self,
        has_lateral_list: list[int],
        batch_size: int,
        frac_lateral: float = 0.4,
        seed: int = 42,
        drop_last: bool = True,
    ):
        if not 0.0 < frac_lateral < 1.0:
            raise ValueError("frac_lateral must be in (0, 1)")

        self.has_lat = np.asarray(has_lateral_list, dtype=np.int64)
        self.bs = int(batch_size)
        self.n_lat = max(1, int(round(self.bs * frac_lateral)))
        self.n_no = self.bs - self.n_lat
        self.drop_last = drop_last

        self.idx_lat = np.where(self.has_lat == 1)[0]
        self.idx_no = np.where(self.has_lat == 0)[0]
        if len(self.idx_lat) == 0:
            raise ValueError("No lateral-present samples available.")
        if len(self.idx_no) == 0:
            raise ValueError("No frontal-only samples available.")

        self.rng = np.random.default_rng(seed)
        n_ref = len(self.has_lat)
        self.num_batches = n_ref // self.bs if drop_last else int(np.ceil(n_ref / self.bs))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            lat = self.rng.choice(self.idx_lat, size=self.n_lat, replace=True)
            replace_no = len(self.idx_no) < self.n_no
            no = self.rng.choice(self.idx_no, size=self.n_no, replace=replace_no)
            batch = np.concatenate([lat, no])
            self.rng.shuffle(batch)
            yield batch.tolist()


def chexpert_collate_with_subgroup(
    demographic_mode: str,
    num_subgroups: int,
    train_diseases: int = DEFAULT_TRAIN_DISEASES,
) -> Callable:
    """Collate for ContextFair: train labels (5 or 13), labels_eval (5), subgroup."""
    mode = demographic_mode.strip().lower()
    n_train = resolve_train_diseases(train_diseases)

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = torch.stack([b["image"] for b in batch], dim=0)
        labels13 = torch.stack([b["labels"] for b in batch], dim=0).float()
        labels_eval = slice_eval_labels(labels13)
        labels = labels_eval if n_train == 5 else labels13
        sex = torch.stack([b["sex"] for b in batch], dim=0).long()
        age = torch.stack([b["age"] for b in batch], dim=0).long()
        if mode == "sex":
            subgroup = sex.clamp(0, num_subgroups - 1)
        elif mode == "age":
            subgroup = age.clamp(0, num_subgroups - 1)
        else:
            raise ValueError(f"demographic_mode must be sex or age, got {demographic_mode!r}")

        return {
            "image": images,
            "labels": labels,
            "labels_eval": labels_eval,
            "sex": sex,
            "age": age,
            "subgroup": subgroup,
            "sample_id": [b["sample_id"] for b in batch],
            "rel_path": [b["rel_path"] for b in batch],
        }

    return _collate


def _dataloader_kwargs(num_workers: int, pin_memory: bool) -> dict[str, Any]:
    kw: dict[str, Any] = {"pin_memory": bool(pin_memory and torch.cuda.is_available())}
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    return kw


def _resolve_split_csvs(data_root: str) -> tuple[str, str, str]:
    train_csv = os.path.join(data_root, "train.csv")
    valid_csv = os.path.join(data_root, "valid.csv")
    test_csv = os.path.join(data_root, "test.csv")
    if not os.path.isfile(test_csv):
        test_csv = valid_csv
    return train_csv, valid_csv, test_csv


def build_frontal_dataloaders(
    data_root: str = DEFAULT_DATA_ROOT,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True,
    aug_style: AugStyle | str = "erm",
    collate_fn: Callable | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train / valid / test loaders using frontal CSV rows only (ERM default)."""
    train_csv, valid_csv, test_csv = _resolve_split_csvs(data_root)

    train_ds = CheXpertFrontalDataset(
        data_root, train_csv, "train", image_size=image_size, train=True, aug_style=aug_style
    )
    valid_ds = CheXpertFrontalDataset(
        data_root, valid_csv, "valid", image_size=image_size, train=False, aug_style="none"
    )
    test_ds = CheXpertFrontalDataset(
        data_root, test_csv, "test", image_size=image_size, train=False, aug_style="none"
    )

    loader_kw = _dataloader_kwargs(num_workers, pin_memory)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        **loader_kw,
    }
    if collate_fn is not None:
        common["collate_fn"] = collate_fn

    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(valid_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def build_chexpert_dataloaders(
    data_root: str = DEFAULT_DATA_ROOT,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    demographic_mode: str = "sex",
    num_subgroups: int = 2,
    train_diseases: int = DEFAULT_TRAIN_DISEASES,
    pin_memory: bool = True,
    aug_style: AugStyle | str = "contextfair",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Frontal loaders with ContextFair subgroup collate (sex or age)."""
    collate = chexpert_collate_with_subgroup(
        demographic_mode, num_subgroups, train_diseases=train_diseases
    )
    return build_frontal_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        pin_memory=pin_memory,
        aug_style=aug_style,
        collate_fn=collate,
    )


def build_multimodal_dataloaders(
    data_root: str = DEFAULT_DATA_ROOT,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True,
    modality_balanced: bool = False,
    frac_lateral: float = 0.4,
    seed: int = 42,
    drop_last: bool = True,
    aug_style: AugStyle | str = "erm",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_csv, valid_csv, test_csv = _resolve_split_csvs(data_root)

    train_ds = CheXpertMultimodalDataset(
        data_root, train_csv, "train", image_size=image_size, train=True, aug_style=aug_style
    )
    valid_ds = CheXpertMultimodalDataset(
        data_root, valid_csv, "valid", image_size=image_size, train=False, aug_style="none"
    )
    test_ds = CheXpertMultimodalDataset(
        data_root, test_csv, "test", image_size=image_size, train=False, aug_style="none"
    )

    if modality_balanced:
        batch_sampler = ModalityBalancedBatchSampler(
            has_lateral_list=train_ds.has_lateral_list,
            batch_size=batch_size,
            frac_lateral=frac_lateral,
            seed=seed,
            drop_last=drop_last,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=bool(pin_memory and torch.cuda.is_available()),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=bool(pin_memory and torch.cuda.is_available()),
        )

    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


def build_dataloaders(
    data_root: str = DEFAULT_DATA_ROOT,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    pin_memory: bool = True,
    frontal_only: bool = True,
    aug_style: AugStyle | str = "erm",
    **kwargs: Any,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Default: frontal-only ERM-style loaders."""
    if frontal_only:
        return build_frontal_dataloaders(
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            pin_memory=pin_memory,
            aug_style=aug_style,
        )
    return build_multimodal_dataloaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        pin_memory=pin_memory,
        aug_style=aug_style,
        **kwargs,
    )


if __name__ == "__main__":
    root = os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)
    if not os.path.isdir(root):
        print(f"DATA_ROOT not found ({root}); skip smoke test.")
    else:
        train, val, test = build_frontal_dataloaders(root, batch_size=2, num_workers=0)
        print(f"train={len(train.dataset)} val={len(val.dataset)} test={len(test.dataset)}")
        b = next(iter(train))
        print({k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in b.items()})
